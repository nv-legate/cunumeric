/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "mapper.h"
#include "legion/legion_mapping.h"
#include "shard.h"
#include <cstdlib>

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {
namespace numpy {

Logger log_numpy("numpy");

//--------------------------------------------------------------------------
NumPyMapper::NumPyMapper(MapperRuntime* rt, Machine m, TaskID first, TaskID last, ShardingID init)
  : Mapper(rt),
    machine(m),
    local_node(get_local_node()),
    total_nodes(get_total_nodes(m)),
    mapper_name(create_name(local_node)),
    first_numpy_task_id(first),
    last_numpy_task_id(last),
    first_sharding_id(init),
    min_gpu_chunk(extract_env("NUMPY_MIN_GPU_CHUNK", 1 << 20, 2)),
    min_cpu_chunk(extract_env("NUMPY_MIN_CPU_CHUNK", 1 << 14, 2)),
    min_omp_chunk(extract_env("NUMPY_MIN_OMP_CHUNK", 1 << 17, 2)),
    eager_fraction(extract_env("NUMPY_EAGER_FRACTION", 16, 1)),
    field_reuse_frac(extract_env("NUMPY_FIELD_REUSE_FRAC", 256, 256)),
    field_reuse_freq(extract_env("NUMPY_FIELD_REUSE_FREQ", 32, 32))
//--------------------------------------------------------------------------
{
  // Query to find all our local processors
  Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  for (Machine::ProcessorQuery::iterator it = local_procs.begin(); it != local_procs.end(); it++) {
    switch (it->kind()) {
      case Processor::LOC_PROC: {
        local_cpus.push_back(*it);
        break;
      }
      case Processor::TOC_PROC: {
        local_gpus.push_back(*it);
        break;
      }
      case Processor::OMP_PROC: {
        local_omps.push_back(*it);
        break;
      }
      case Processor::IO_PROC: {
        local_ios.push_back(*it);
        break;
      }
      case Processor::PY_PROC: {
        local_pys.push_back(*it);
        break;
      }
      default: break;
    }
  }
  // Now do queries to find all our local memories
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (std::vector<Processor>::const_iterator it = local_gpus.begin(); it != local_gpus.end();
       it++) {
    Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(*it);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[*it] = local_framebuffer.first();
  }
  for (std::vector<Processor>::const_iterator it = local_omps.begin(); it != local_omps.end();
       it++) {
    Machine::MemoryQuery local_numa(machine);
    local_numa.local_address_space();
    local_numa.only_kind(Memory::SOCKET_MEM);
    local_numa.best_affinity_to(*it);
    if (local_numa.count() > 0)  // if we have NUMA memories then use them
      local_numa_domains[*it] = local_numa.first();
    else  // Otherwise we just use the local system memory
      local_numa_domains[*it] = local_system_memory;
  }
}

//--------------------------------------------------------------------------
NumPyMapper::NumPyMapper(const NumPyMapper& rhs)
  : Mapper(rhs.runtime),
    machine(rhs.machine),
    local_node(0),
    total_nodes(0),
    mapper_name(NULL),
    first_numpy_task_id(0),
    last_numpy_task_id(0),
    first_sharding_id(0),
    min_gpu_chunk(0),
    min_cpu_chunk(0),
    min_omp_chunk(0),
    eager_fraction(0),
    field_reuse_frac(0),
    field_reuse_freq(0)
//--------------------------------------------------------------------------
{
  // should never be called
  LEGATE_ABORT;
}

//--------------------------------------------------------------------------
NumPyMapper::~NumPyMapper(void)
//--------------------------------------------------------------------------
{
  free(const_cast<char*>(mapper_name));
  // Compute the size of all our remaining instances in each memory
  const char* show_usage = getenv("NUMPY_SHOW_USAGE");
  if (show_usage != NULL) {
    std::map<Memory, size_t> mem_sizes;
    for (std::map<FieldMemInfo, InstanceInfos>::const_iterator lit = local_instances.begin();
         lit != local_instances.end();
         lit++) {
      for (std::vector<InstanceInfo>::const_iterator it = lit->second.instances.begin();
           it != lit->second.instances.end();
           it++) {
        const size_t inst_size                    = it->instance.get_instance_size();
        std::map<Memory, size_t>::iterator finder = mem_sizes.find(lit->first.memory);
        if (finder == mem_sizes.end())
          mem_sizes[lit->first.memory] = inst_size;
        else
          finder->second += inst_size;
      }
    }
    const char* memory_kinds[] = {
#define MEM_NAMES(name, desc) desc,
      REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
    };
    for (std::map<Memory, size_t>::const_iterator it = mem_sizes.begin(); it != mem_sizes.end();
         it++) {
      const size_t capacity = it->first.capacity();
      log_numpy.print(
        "Legate.NumPy used %ld bytes of %s memory %llx with "
        "%ld total bytes (%.2g%%)",
        it->second,
        memory_kinds[it->first.kind()],
        it->first.id,
        capacity,
        100.0 * double(it->second) / capacity);
    }
  }
}

//--------------------------------------------------------------------------
NumPyMapper& NumPyMapper::operator=(const NumPyMapper& rhs)
//--------------------------------------------------------------------------
{
  // should never be called
  LEGATE_ABORT
  return *this;
}

//--------------------------------------------------------------------------
/*static*/ AddressSpace NumPyMapper::get_local_node(void)
//--------------------------------------------------------------------------
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

//--------------------------------------------------------------------------
/*static*/ size_t NumPyMapper::get_total_nodes(Machine m)
//--------------------------------------------------------------------------
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::set<AddressSpace> spaces;
  for (Machine::ProcessorQuery::iterator it = query.begin(); it != query.end(); it++)
    spaces.insert(it->address_space());
  return spaces.size();
}

//--------------------------------------------------------------------------
/*static*/ const char* NumPyMapper::create_name(AddressSpace node)
//--------------------------------------------------------------------------
{
  char buffer[128];
  snprintf(buffer, 127, "NumPy Mapper on Node %d", node);
  return strdup(buffer);
}

//--------------------------------------------------------------------------
const char* NumPyMapper::get_mapper_name(void) const
//--------------------------------------------------------------------------
{
  return mapper_name;
}

//--------------------------------------------------------------------------
Mapper::MapperSyncModel NumPyMapper::get_mapper_sync_model(void) const
//--------------------------------------------------------------------------
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

//--------------------------------------------------------------------------
void NumPyMapper::select_task_options(const MapperContext ctx,
                                      const Task& task,
                                      TaskOptions& output)
//--------------------------------------------------------------------------
{
  assert(task.get_depth() > 0);
  if (!local_gpus.empty() && has_variant(ctx, task, Processor::TOC_PROC))
    output.initial_proc = local_gpus.front();
  else if (!local_omps.empty() && has_variant(ctx, task, Processor::OMP_PROC))
    output.initial_proc = local_omps.front();
  else
    output.initial_proc = local_cpus.front();
  // We never want valid instances
  output.valid_instances = false;
}

//--------------------------------------------------------------------------
void NumPyMapper::premap_task(const MapperContext ctx,
                              const Task& task,
                              const PremapTaskInput& input,
                              PremapTaskOutput& output)
//--------------------------------------------------------------------------
{
  // NO-op since we know that all our futures should be mapped in the system memory
}

//--------------------------------------------------------------------------
void NumPyMapper::slice_task(const MapperContext ctx,
                             const Task& task,
                             const SliceTaskInput& input,
                             SliceTaskOutput& output)
//--------------------------------------------------------------------------
{
  // For multi-node cases we should already have been sharded so we
  // should just have one or a few points here on this node, so iterate
  // them and round-robin them across the local processors here
  output.slices.reserve(input.domain.get_volume());
  // Get the sharding functor for this operation and then use it to localize
  // the points onto the processors of this shard
  // const ShardingID sid          = select_sharding_functor(ctx, task);
  // NumPyShardingFunctor* functor = find_sharding_functor(sid);
  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);
  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        const unsigned local_index = 0;
        // functor->localize(itr.p, sharding_domain, total_nodes, local_node) % local_cpus.size();
        output.slices.push_back(TaskSlice(
          Domain(itr.p, itr.p), local_cpus[local_index], false /*recurse*/, false /*stealable*/));
      }
      break;
    }
    case Processor::TOC_PROC: {
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        const unsigned local_index = 0;
        // functor->localize(itr.p, sharding_domain, total_nodes, local_node) % local_gpus.size();
        output.slices.push_back(TaskSlice(
          Domain(itr.p, itr.p), local_gpus[local_index], false /*recurse*/, false /*stealable*/));
      }
      break;
    }
    case Processor::OMP_PROC: {
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        const unsigned local_index = 0;
        // functor->localize(itr.p, sharding_domain, total_nodes, local_node) % local_omps.size();
        output.slices.push_back(TaskSlice(
          Domain(itr.p, itr.p), local_omps[local_index], false /*recurse*/, false /*stealable*/));
      }
      break;
    }
    default: LEGATE_ABORT
  }
}

//--------------------------------------------------------------------------
bool NumPyMapper::has_variant(const MapperContext ctx, const Task& task, Processor::Kind kind)
//--------------------------------------------------------------------------
{
  const std::pair<TaskID, Processor::Kind> key(task.task_id, kind);
  // Check to see if we already have it
  std::map<std::pair<TaskID, Processor::Kind>, VariantID>::const_iterator finder =
    leaf_variants.find(key);
  if ((finder != leaf_variants.end()) && (finder->second != 0)) return true;
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, key.first, variants, key.second);
  // Process all the results, record if we found what we were looking for
  bool has_leaf  = false;
  bool has_inner = false;
  for (std::vector<VariantID>::const_iterator it = variants.begin(); it != variants.end(); it++) {
    assert((*it) > 0);
    switch (*it) {
      case LEGATE_CPU_VARIANT:
      case LEGATE_OMP_VARIANT:
      case LEGATE_GPU_VARIANT: {
        has_leaf           = true;
        leaf_variants[key] = *it;
        break;
      }
      case LEGATE_SUB_CPU_VARIANT:
      case LEGATE_SUB_GPU_VARIANT:
      case LEGATE_SUB_OMP_VARIANT: {
        has_inner           = true;
        inner_variants[key] = *it;
        break;
      }
      default:        // TODO: handle vectorized variants
        LEGATE_ABORT  // unhandled variant kind
    }
  }
  if (!has_leaf) leaf_variants[key] = 0;
  if (!has_inner) inner_variants[key] = 0;
  return has_leaf;
}

//--------------------------------------------------------------------------
VariantID NumPyMapper::find_variant(const MapperContext ctx, const Task& task, bool subrank)
//--------------------------------------------------------------------------
{
  return find_variant(ctx, task, subrank, task.target_proc);
}

//--------------------------------------------------------------------------
VariantID NumPyMapper::find_variant(const MapperContext ctx,
                                    const Task& task,
                                    bool subrank,
                                    Processor target_proc)
//--------------------------------------------------------------------------
{
  const std::pair<TaskID, Processor::Kind> key(task.task_id, target_proc.kind());
  if (subrank) {
    std::map<std::pair<TaskID, Processor::Kind>, VariantID>::const_iterator finder =
      inner_variants.find(key);
    if ((finder != inner_variants.end()) && (finder->second != 0)) return finder->second;
  } else {
    std::map<std::pair<TaskID, Processor::Kind>, VariantID>::const_iterator finder =
      leaf_variants.find(key);
    if ((finder != leaf_variants.end()) && (finder->second != 0)) return finder->second;
  }
  // Haven't seen it before so let's look it up to make sure it exists
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, key.first, variants, key.second);
  VariantID result = 0;  // 0 is reserved
  bool has_leaf    = false;
  bool has_inner   = false;
  // Process all the results, record if we found what we were looking for
  for (std::vector<VariantID>::const_iterator it = variants.begin(); it != variants.end(); it++) {
    assert((*it) > 0);
    switch (*it) {
      case LEGATE_CPU_VARIANT:
      case LEGATE_OMP_VARIANT:
      case LEGATE_GPU_VARIANT: {
        has_leaf           = true;
        leaf_variants[key] = *it;
        if (!subrank) result = *it;
        break;
      }
      case LEGATE_SUB_CPU_VARIANT:
      case LEGATE_SUB_GPU_VARIANT:
      case LEGATE_SUB_OMP_VARIANT: {
        has_inner           = true;
        inner_variants[key] = *it;
        if (subrank) result = *it;
        break;
      }
      default:        // TODO: handle vectorized variants
        LEGATE_ABORT  // unhandled variant kind
    }
  }
  if (!has_leaf) leaf_variants[key] = 0;
  if (!has_inner) inner_variants[key] = 0;
  // We must always be able to find the variant;
  assert(result != 0);
  return result;
}

//--------------------------------------------------------------------------
void NumPyMapper::map_task(const MapperContext ctx,
                           const Task& task,
                           const MapTaskInput& input,
                           MapTaskOutput& output)
//--------------------------------------------------------------------------
{
  // Should never be mapping the top-level task here
  assert(task.get_depth() > 0);
  // This is one of our normal Legate tasks
  // First let's see if this is sub-rankable
  output.chosen_instances.resize(task.regions.size());
  // We've subsumed the tag for now to capture sharding function IDs
#if 0
  if (task.tag & NUMPY_SUBRANKABLE_TAG) {
    output.chosen_variant = find_variant(ctx, task, true /*subrank*/);
    // Request virtual mappings for all the region requirements
    const PhysicalInstance virt = PhysicalInstance::get_virtual_instance();
    for (unsigned idx = 0; idx < task.regions.size(); idx++)
      output.chosen_instances[idx].push_back(virt);
    return;
  } else    // Not subrankable so get the non-subrank variant
#endif
  output.chosen_variant = find_variant(ctx, task, false /*subrank*/);
  // Normal task and not sub-rankable, so let's actually do the mapping
  Memory target_memory = Memory::NO_MEMORY;
  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      target_memory = local_system_memory;
      break;
    }
    case Processor::TOC_PROC: {
      target_memory = local_frame_buffers[task.target_proc];
      break;
    }
    case Processor::OMP_PROC: {
      target_memory = local_numa_domains[task.target_proc];
      break;
    }
    default: LEGATE_ABORT
  }
  // Map each field separately for each of the logical regions
  std::vector<PhysicalInstance> needed_acquires;
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    const RegionRequirement& req = task.regions[idx];
    // Skip any regions that have been projected out
    if (!req.region.exists()) continue;
    std::vector<PhysicalInstance>& instances = output.chosen_instances[idx];
    // Get the reference to our valid instances in case we decide to use them
    const std::vector<PhysicalInstance>& valid = input.valid_instances[idx];
    instances.resize(req.privilege_fields.size());
    unsigned index = 0;
    const bool memoize =
      ((req.tag & NUMPY_NO_MEMOIZE_TAG) == 0) && (req.privilege != LEGION_REDUCE);
    for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
         it != req.privilege_fields.end();
         it++, index++)
      if (map_numpy_array(ctx,
                          task,
                          idx,
                          req.region,
                          *it,
                          target_memory,
                          task.target_proc,
                          valid,
                          instances[index],
                          memoize,
                          req.redop))
        needed_acquires.push_back(instances[index]);
  }
  // Do an acquire on all the instances so we have our result
  // Keep doing this until we succed or we get an out of memory error
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    // If we failed to acquire any of the instances we need to prune them
    // out of the mapper's data structure so do that first
    std::set<PhysicalInstance> failed_acquires;
    filter_failed_acquires(needed_acquires, failed_acquires);
    // Now go through all our region requirements and and figure out which
    // region requirements and fields need to attempt to remap
    for (unsigned idx1 = 0; idx1 < task.regions.size(); idx1++) {
      const RegionRequirement& req = task.regions[idx1];
      // Skip any regions that have been projected out
      if (!req.region.exists()) continue;
      std::vector<PhysicalInstance>& instances = output.chosen_instances[idx1];
      std::set<FieldID>::const_iterator fit    = req.privilege_fields.begin();
      for (unsigned idx2 = 0; idx2 < instances.size(); idx2++, fit++) {
        if (failed_acquires.find(instances[idx2]) == failed_acquires.end()) continue;
        // Now try to remap it
        const FieldID fid                          = *fit;
        const std::vector<PhysicalInstance>& valid = input.valid_instances[idx1];
        const bool memoize =
          ((req.tag & NUMPY_NO_MEMOIZE_TAG) == 0) && (req.privilege != LEGION_REDUCE);
        if (map_numpy_array(ctx,
                            task,
                            idx1,
                            req.region,
                            fid,
                            target_memory,
                            task.target_proc,
                            valid,
                            instances[idx2],
                            memoize,
                            req.redop))
          needed_acquires.push_back(instances[idx2]);
      }
    }
  }
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);
}

//--------------------------------------------------------------------------
void NumPyMapper::map_replicate_task(const MapperContext ctx,
                                     const Task& task,
                                     const MapTaskInput& input,
                                     const MapTaskOutput& def_output,
                                     MapReplicateTaskOutput& output)
//--------------------------------------------------------------------------
{
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
bool NumPyMapper::find_existing_instance(LogicalRegion region,
                                         FieldID fid,
                                         Memory target_memory,
                                         PhysicalInstance& result,
                                         Strictness strictness)
//--------------------------------------------------------------------------
{
  // See if we already have it in our local instances
  const FieldMemInfo info(region.get_tree_id(), fid, target_memory);
  std::map<FieldMemInfo, InstanceInfos>::const_iterator finder = local_instances.find(info);
  if ((finder != local_instances.end()) && finder->second.has_instance(region, result)) {
    return true;
  } else if (strictness == Strictness::strict) {
    return false;
  }
  // See if we can find an existing instance in any memory
  const FieldMemInfo info_sysmem(region.get_tree_id(), fid, local_system_memory);
  finder = local_instances.find(info_sysmem);
  if ((finder != local_instances.end()) && finder->second.has_instance(region, result)) {
    return true;
  }
  for (std::map<Processor, Memory>::const_iterator it = local_frame_buffers.begin();
       it != local_frame_buffers.end();
       it++) {
    const FieldMemInfo info_fb(region.get_tree_id(), fid, it->second);
    finder = local_instances.find(info_fb);
    if ((finder != local_instances.end()) && finder->second.has_instance(region, result)) {
      return true;
    }
  }
  for (std::map<Processor, Memory>::const_iterator it = local_numa_domains.begin();
       it != local_numa_domains.end();
       it++) {
    const FieldMemInfo info_numa(region.get_tree_id(), fid, it->second);
    finder = local_instances.find(info_numa);
    if ((finder != local_instances.end()) && finder->second.has_instance(region, result)) {
      return true;
    }
  }
  return false;
}

//--------------------------------------------------------------------------
bool NumPyMapper::map_numpy_array(const MapperContext ctx,
                                  const Mappable& mappable,
                                  unsigned index,
                                  LogicalRegion region,
                                  FieldID fid,
                                  Memory target_memory,
                                  Processor target_proc,
                                  const std::vector<PhysicalInstance>& valid,
                                  PhysicalInstance& result,
                                  bool memoize_result,
                                  ReductionOpID redop /*=0*/)
//--------------------------------------------------------------------------
{
  // If we're making a reduction instance, we should just make it now
  if (redop != 0) {
    // Switch the target memory if we're going to a GPU because
    // Realm's DMA system still does not support reductions
    if (target_memory.kind() == Memory::GPU_FB_MEM) target_memory = local_zerocopy_memory;
    const std::vector<LogicalRegion> regions(1, region);
    LayoutConstraintSet layout_constraints;
    // No specialization
    layout_constraints.add_constraint(SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, redop));
    // SOA-C dimension ordering
    std::vector<DimensionKind> dimension_ordering(4);
    dimension_ordering[0] = DIM_Z;
    dimension_ordering[1] = DIM_Y;
    dimension_ordering[2] = DIM_X;
    dimension_ordering[3] = DIM_F;
    layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, false /*contiguous*/));
    // Constraint for the kind of memory
    layout_constraints.add_constraint(MemoryConstraint(target_memory.kind()));
    // Make sure we have our field
    const std::vector<FieldID> fields(1, fid);
    layout_constraints.add_constraint(FieldConstraint(fields, true /*contiguous*/));
    if (!runtime->create_physical_instance(
          ctx, target_memory, layout_constraints, regions, result, true /*acquire*/))
      report_failed_mapping(mappable, index, target_memory, redop);
    // We already did the acquire
    return false;
  }
  // See if we already have it in our local instances
  const FieldMemInfo info_key(region.get_tree_id(), fid, target_memory);
  std::map<FieldMemInfo, InstanceInfos>::const_iterator finder = local_instances.find(info_key);
  if ((finder != local_instances.end()) && finder->second.has_instance(region, result)) {
    // Needs acquire to keep the runtime happy
    return true;
  }
  // There's a little asymmetry here between CPUs and GPUs for NUMA effects
  // For CPUs NUMA-effects are within a factor of 2X additional latency and
  // reduced bandwidth, so it's better to just use data where it is rather
  // than move it. For GPUs though, the difference between local framebuffer
  // and remote can be on the order of 800 GB/s versus 20 GB/s over NVLink
  // so it's better to move things local, so we'll always try to make a local
  // instance before checking for a nearby instance in a different GPU.
  if (target_proc.exists() && ((target_proc.kind() == Processor::LOC_PROC) ||
                               (target_proc.kind() == Processor::OMP_PROC))) {
    Machine::MemoryQuery affinity_mems(machine);
    affinity_mems.has_affinity_to(target_proc);
    for (Machine::MemoryQuery::iterator it = affinity_mems.begin(); it != affinity_mems.end();
         it++) {
      const FieldMemInfo affinity_info(region.get_tree_id(), fid, *it);
      finder = local_instances.find(affinity_info);
      if ((finder != local_instances.end()) && finder->second.has_instance(region, result))
        // Needs acquire to keep the runtime happy
        return true;
    }
  }
  // Haven't made this instance before, so make it now
  // We can do an interesting optimization here to try to reduce unnecessary
  // inter-memory copies. For logical regions that are overlapping we try
  // to accumulate as many as possible into one physical instance and use
  // that instance for all the tasks for the different regions.
  // First we have to see if there is anything we overlap with
  const IndexSpace is = region.get_index_space();
  // This whole process has to appear atomic
  runtime->disable_reentrant(ctx);
  InstanceInfos& infos = local_instances[info_key];
  // One more check once we get the lock
  if (infos.has_instance(region, result)) {
    runtime->enable_reentrant(ctx);
    return true;
  }
  const Domain dom = runtime->get_index_space_domain(ctx, is);
  std::vector<unsigned> overlaps;
  // Regions to include in the overlap from other fields
  std::set<LogicalRegion> other_field_overlaps;
  // This is guaranteed to be a rectangle
  Domain upper_bound;
  switch (is.get_dim()) {
#define LEGATE_DIMFUNC(DN)                                                                        \
  case DN: {                                                                                      \
    bool changed   = false;                                                                       \
    Rect<DN> bound = dom.bounds<DN, coord_t>();                                                   \
    for (unsigned idx = 0; idx < infos.instances.size(); idx++) {                                 \
      const InstanceInfo& info = infos.instances[idx];                                            \
      Rect<DN> other           = info.bounding_box;                                               \
      Rect<DN> intersect       = bound.intersection(other);                                       \
      if (intersect.empty()) continue;                                                            \
      /*Don't merge if the unused space would be more than the space saved*/                      \
      Rect<DN> union_bbox = bound.union_bbox(other);                                              \
      size_t bound_volume = bound.volume();                                                       \
      size_t union_volume = union_bbox.volume();                                                  \
      /* If it didn't get any bigger then we can keep going*/                                     \
      if (bound_volume == union_volume) continue;                                                 \
      size_t intersect_volume = intersect.volume();                                               \
      /* Only allow merging if it isn't "too big"*/                                               \
      /* We define "too big" as the size of the "unused" points being bigger than the             \
       * intersection*/                                                                           \
      if ((union_volume - (bound_volume + other.volume() - intersect_volume)) > intersect_volume) \
        continue;                                                                                 \
      overlaps.push_back(idx);                                                                    \
      bound   = union_bbox;                                                                       \
      changed = true;                                                                             \
    }                                                                                             \
    /* If we didn't find any overlapping modifications check adjacent fields in the same tree*/   \
    /* to see if we can use them to infer what our shape should be.*/                             \
    if (!changed) {                                                                               \
      for (std::map<FieldMemInfo, InstanceInfos>::const_iterator it = local_instances.begin();    \
           it != local_instances.end();                                                           \
           it++) {                                                                                \
        if ((it->first.tid != info_key.tid) || (it->first.fid == info_key.fid) ||                 \
            (it->first.memory != info_key.memory))                                                \
          continue;                                                                               \
        std::map<LogicalRegion, unsigned>::const_iterator finder =                                \
          it->second.region_mapping.find(region);                                                 \
        if (finder != it->second.region_mapping.end()) {                                          \
          const InstanceInfo& other_info = it->second.instances[finder->second];                  \
          Rect<DN> other                 = other_info.bounding_box;                               \
          bound                          = bound.union_bbox(other);                               \
          other_field_overlaps.insert(other_info.regions.begin(), other_info.regions.end());      \
        }                                                                                         \
      }                                                                                           \
    }                                                                                             \
    upper_bound = Domain(bound);                                                                  \
    break;                                                                                        \
  }
    LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
    default: assert(false);
  }
  // We're going to need some of this constraint information no matter
  // which path we end up taking below
  LayoutConstraintSet layout_constraints;
  // No specialization
  layout_constraints.add_constraint(SpecializedConstraint());
  // SOA-C dimension ordering
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_Z;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_X;
  dimension_ordering[3] = DIM_F;
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, false /*contiguous*/));
  // Constraint for the kind of memory
  layout_constraints.add_constraint(MemoryConstraint(target_memory.kind()));
  // Make sure we have our field
  const std::vector<FieldID> fields(1, fid);
  layout_constraints.add_constraint(FieldConstraint(fields, true /*contiguous*/));
  // Check to see if we have any overlaps
  if (overlaps.empty()) {
    // No overlaps, so just go ahead and make our instance and add it
    std::vector<LogicalRegion> regions(1, region);
    // If we're bringing in other regions include them as well in this set
    if (!other_field_overlaps.empty()) {
      other_field_overlaps.erase(region);
      regions.insert(regions.end(), other_field_overlaps.begin(), other_field_overlaps.end());
    }
    bool created;
    size_t footprint;
    if (runtime->find_or_create_physical_instance(ctx,
                                                  target_memory,
                                                  layout_constraints,
                                                  regions,
                                                  result,
                                                  created,
                                                  true /*acquire*/,
                                                  memoize_result ? GC_NEVER_PRIORITY : 0,
                                                  false /*tight bounds*/,
                                                  &footprint)) {
      // We succeeded in making the instance where we want it
      assert(result.exists());
      if (created)
        log_numpy.info("%s created instance %lx containing %zd bytes in memory " IDFMT,
                       mapper_name,
                       result.get_instance_id(),
                       footprint,
                       target_memory.id);
      // Only save the result for future use if it is not an external instance
      if (memoize_result && !result.is_external_instance()) {
        const unsigned idx = infos.insert(region, upper_bound, result);
        InstanceInfo& info = infos.instances[idx];
        for (std::set<LogicalRegion>::const_iterator it = other_field_overlaps.begin();
             it != other_field_overlaps.end();
             it++) {
          if ((*it) == region) continue;
          infos.region_mapping[*it] = idx;
          info.regions.push_back(*it);
        }
      }
      // We made it so no need for an acquire
      runtime->enable_reentrant(ctx);
      return false;
    }

  } else if (overlaps.size() == 1) {
    // Overlap with exactly one other instance
    InstanceInfo& info = infos.instances[overlaps[0]];
    // A Legion bug prevents us from doing this case
    if (info.bounding_box == upper_bound) {
      // Easy case of dominance, so just add it
      info.regions.push_back(region);
      infos.region_mapping[region] = overlaps[0];
      result                       = info.instance;
      runtime->enable_reentrant(ctx);
      // Didn't make it so we need to acquire it
      return true;
    } else {
      // We have to make a new instance
      info.regions.push_back(region);
      bool created;
      size_t footprint;
      if (runtime->find_or_create_physical_instance(ctx,
                                                    target_memory,
                                                    layout_constraints,
                                                    info.regions,
                                                    result,
                                                    created,
                                                    true /*acquire*/,
                                                    GC_NEVER_PRIORITY,
                                                    false /*tight bounds*/,
                                                    &footprint)) {
        // We succeeded in making the instance where we want it
        assert(result.exists());
        if (created)
          log_numpy.info("%s created instance %lx containing %zd bytes in memory " IDFMT,
                         mapper_name,
                         result.get_instance_id(),
                         footprint,
                         target_memory.id);
        // Remove the GC priority on the old instance back to 0
        runtime->set_garbage_collection_priority(ctx, info.instance, 0);
        // Update everything in place
        info.instance                = result;
        info.bounding_box            = upper_bound;
        infos.region_mapping[region] = overlaps[0];
        runtime->enable_reentrant(ctx);
        // We made it so no need for an acquire
        return false;
      } else  // Failed to make it so pop the logical region name back off
        info.regions.pop_back();
    }
  } else {
    // Overlap with multiple previous instances
    std::vector<LogicalRegion> combined_regions(1, region);
    for (std::vector<unsigned>::const_iterator it = overlaps.begin(); it != overlaps.end(); it++)
      combined_regions.insert(combined_regions.end(),
                              infos.instances[*it].regions.begin(),
                              infos.instances[*it].regions.end());
    // Try to make it
    bool created;
    size_t footprint;
    if (runtime->find_or_create_physical_instance(ctx,
                                                  target_memory,
                                                  layout_constraints,
                                                  combined_regions,
                                                  result,
                                                  created,
                                                  true /*acquire*/,
                                                  GC_NEVER_PRIORITY,
                                                  false /*tight bounds*/,
                                                  &footprint)) {
      // We succeeded in making the instance where we want it
      assert(result.exists());
      if (created)
        log_numpy.info("%s created instance %lx containing %zd bytes in memory " IDFMT,
                       mapper_name,
                       result.get_instance_id(),
                       footprint,
                       target_memory.id);
      // Remove all the previous entries back to front
      for (std::vector<unsigned>::const_reverse_iterator it = overlaps.crbegin();
           it != overlaps.crend();
           it++) {
        // Remove the GC priority on the old instance
        runtime->set_garbage_collection_priority(ctx, infos.instances[*it].instance, 0);
        infos.erase(*it);
      }
      // Add the new entry
      const unsigned index = infos.instances.size();
      infos.instances.resize(index + 1);
      InstanceInfo& info = infos.instances[index];
      info.instance      = result;
      info.bounding_box  = upper_bound;
      info.regions       = combined_regions;
      // Update the mappings for all the instances
      // This really sucks but it should be pretty rare
      // We can start at the entry of the first overlap since everything
      // before that is guaranteed to be unchanged
      for (unsigned idx = overlaps[0]; idx < infos.instances.size(); idx++) {
        for (std::vector<LogicalRegion>::const_iterator it = infos.instances[idx].regions.begin();
             it != infos.instances[idx].regions.end();
             it++)
          infos.region_mapping[*it] = idx;
      }
      runtime->enable_reentrant(ctx);
      // We made it so no need for an acquire
      return false;
    }
  }
  // Done with the atomic part
  runtime->enable_reentrant(ctx);
  // If we get here it's because we failed to make the instance, we still
  // have a few more tricks that we can try
  // First see if we can find an existing valid instance that we can use
  // with affinity to our target processor
  if (!valid.empty()) {
    for (std::vector<PhysicalInstance>::const_iterator it = valid.begin(); it != valid.end();
         it++) {
      // If it doesn't have the field then we don't care
      if (!it->has_field(fid)) continue;
      if (!target_proc.exists() || machine.has_affinity(target_proc, it->get_location())) {
        result = *it;
        return true;
      }
    }
  }
  // Still couldn't find an instance, see if we can find any instances
  // in memories that are local to our node that we can use
  if (target_proc.exists()) {
    Machine::MemoryQuery affinity_mems(machine);
    affinity_mems.has_affinity_to(target_proc);
    for (Machine::MemoryQuery::iterator it = affinity_mems.begin(); it != affinity_mems.end();
         it++) {
      const FieldMemInfo affinity_info(region.get_tree_id(), fid, *it);
      finder = local_instances.find(affinity_info);
      if ((finder != local_instances.end()) && finder->second.has_instance(region, result))
        // Needs acquire to keep the runtime happy
        return true;
    }
  } else if (find_existing_instance(region, fid, target_memory, result)) {
    return true;
  }
  // If we make it here then we failed entirely
  report_failed_mapping(mappable, index, target_memory, redop);
  return true;
}

//--------------------------------------------------------------------------
void NumPyMapper::filter_failed_acquires(std::vector<PhysicalInstance>& needed_acquires,
                                         std::set<PhysicalInstance>& failed_acquires)
//--------------------------------------------------------------------------
{
  for (std::vector<PhysicalInstance>::const_iterator it = needed_acquires.begin();
       it != needed_acquires.end();
       it++) {
    if (failed_acquires.find(*it) != failed_acquires.end()) continue;
    failed_acquires.insert(*it);
    const Memory mem       = it->get_location();
    const RegionTreeID tid = it->get_tree_id();
    for (std::map<FieldMemInfo, InstanceInfos>::iterator fit = local_instances.begin();
         fit != local_instances.end();
         /*nothing*/) {
      if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
        fit++;
        continue;
      }
      if (fit->second.filter(*it)) {
        std::map<FieldMemInfo, InstanceInfos>::iterator to_delete = fit++;
        local_instances.erase(to_delete);
      } else
        fit++;
    }
  }
  needed_acquires.clear();
}

//--------------------------------------------------------------------------
void NumPyMapper::report_failed_mapping(const Mappable& mappable,
                                        unsigned index,
                                        Memory target_memory,
                                        ReductionOpID redop)
//--------------------------------------------------------------------------
{
  const char* memory_kinds[] = {
#define MEM_NAMES(name, desc) desc,
    REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
  };
  switch (mappable.get_mappable_type()) {
    case Mappable::TASK_MAPPABLE: {
      const Task* task = mappable.as_task();
      if (redop > 0)
        log_numpy.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of task %s (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          task->get_task_name(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        log_numpy.error(
          "Mapper %s failed to map region requirement %d of "
          "task %s (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          task->get_task_name(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::COPY_MAPPABLE: {
      if (redop > 0)
        log_numpy.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of copy (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        log_numpy.error(
          "Mapper %s failed to map region requirement %d of "
          "copy (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::INLINE_MAPPABLE: {
      if (redop > 0)
        log_numpy.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of inline mapping (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        log_numpy.error(
          "Mapper %s failed to map region requirement %d of "
          "inline mapping (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::PARTITION_MAPPABLE: {
      assert(redop == 0);
      log_numpy.error(
        "Mapper %s failed to map region requirement %d of "
        "partition (UID %lld) into %s memory " IDFMT,
        get_mapper_name(),
        index,
        mappable.get_unique_id(),
        memory_kinds[target_memory.kind()],
        target_memory.id);
      break;
    }
    default: LEGATE_ABORT  // should never get here
  }
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_task_variant(const MapperContext ctx,
                                      const Task& task,
                                      const SelectVariantInput& input,
                                      SelectVariantOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_variant = find_variant(ctx, task, false /*subrank*/, input.processor);
}

//--------------------------------------------------------------------------
void NumPyMapper::postmap_task(const MapperContext ctx,
                               const Task& task,
                               const PostMapInput& input,
                               PostMapOutput& output)
//--------------------------------------------------------------------------
{
  // We should currently never get this call in Legate
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_task_sources(const MapperContext ctx,
                                      const Task& task,
                                      const SelectTaskSrcInput& input,
                                      SelectTaskSrcOutput& output)
//--------------------------------------------------------------------------
{
  numpy_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void NumPyMapper::numpy_select_sources(const MapperContext ctx,
                                       const PhysicalInstance& target,
                                       const std::vector<PhysicalInstance>& sources,
                                       std::deque<PhysicalInstance>& ranking)
//--------------------------------------------------------------------------
{
  std::map<Memory, unsigned /*bandwidth*/> source_memories;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination, we'll only rank sources from the
  // local node if there are any
  bool all_local = false;
  // TODO: consider layouts when ranking source to help out the DMA system
  Memory destination_memory = target.get_location();
  std::vector<MemoryMemoryAffinity> affinity(1);
  // fill in a vector of the sources with their bandwidths and sort them
  std::vector<std::pair<PhysicalInstance, unsigned /*bandwidth*/>> band_ranking;
  for (unsigned idx = 0; idx < sources.size(); idx++) {
    const PhysicalInstance& instance = sources[idx];
    Memory location                  = instance.get_location();
    if (location.address_space() == local_node) {
      if (!all_local) {
        source_memories.clear();
        band_ranking.clear();
        all_local = true;
      }
    } else if (all_local)  // Skip any remote instances once we're local
      continue;
    std::map<Memory, unsigned>::const_iterator finder = source_memories.find(location);
    if (finder == source_memories.end()) {
      affinity.clear();
      machine.get_mem_mem_affinity(
        affinity, location, destination_memory, false /*not just local affinities*/);
      unsigned memory_bandwidth = 0;
      if (!affinity.empty()) {
        assert(affinity.size() == 1);
        memory_bandwidth = affinity[0].bandwidth;
#if 0
          } else {
            // TODO: More graceful way of dealing with multi-hop copies
            log_numpy.warning("Legate mapper is potentially "
                              "requesting a multi-hop copy between memories "
                              IDFMT " and " IDFMT "!", location.id,
                              destination_memory.id);
#endif
      }
      source_memories[location] = memory_bandwidth;
      band_ranking.push_back(std::pair<PhysicalInstance, unsigned>(instance, memory_bandwidth));
    } else
      band_ranking.push_back(std::pair<PhysicalInstance, unsigned>(instance, finder->second));
  }
  assert(!band_ranking.empty());
  // Easy case of only one instance
  if (band_ranking.size() == 1) {
    ranking.push_back(band_ranking.begin()->first);
    return;
  }
  // Sort them by bandwidth
  std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
  // Iterate from largest bandwidth to smallest
  for (std::vector<std::pair<PhysicalInstance, unsigned>>::const_reverse_iterator it =
         band_ranking.rbegin();
       it != band_ranking.rend();
       it++)
    ranking.push_back(it->first);
}

//--------------------------------------------------------------------------
void NumPyMapper::speculate(const MapperContext ctx, const Task& task, SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const Task& task,
                                   const TaskProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // Shouldn't get any profiling feedback currently
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
/*static*/ int NumPyMapper::find_key_region(const Task& task)
//--------------------------------------------------------------------------
{
  for (unsigned idx = 0; idx < task.regions.size(); idx++)
    if (task.regions[idx].tag & NUMPY_KEY_REGION_TAG) return idx;
  return -1;
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Task& task,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = 0;  // select_sharding_functor(ctx, task);
}

//--------------------------------------------------------------------------
ShardingID NumPyMapper::select_sharding_functor(const MapperContext ctx, const Task& task)
//--------------------------------------------------------------------------
{
  const int launch_dim = task.index_domain.get_dim();
  if (task.regions.empty()) {
    switch (launch_dim) {
      case 1: return first_sharding_id + NUMPY_SHARD_TILE_1D;
      case 2: return first_sharding_id + NUMPY_SHARD_TILE_2D;
      case 3: return first_sharding_id + NUMPY_SHARD_TILE_3D;
      default: LEGATE_ABORT
    }
    // keep the compiler happy
    return 0;
  }
  // Decode the task and see if it is an "interesting" task
  NumPyOpCode op_code = decode_task_id(task.task_id);
  if (op_code == NumPyOpCode::NUMPY_DOT || op_code == NumPyOpCode::NUMPY_MATVECMUL ||
      op_code == NumPyOpCode::NUMPY_MATMUL) {
    switch (launch_dim) {
      case 1:  // vector dot product
        return first_sharding_id + NUMPY_SHARD_TILE_1D;
      case 2:  // GEMV
        return first_sharding_id + NUMPY_SHARD_TILE_2D;
      case 3:  // GEMM
      {
        // Find the key region
        const int key_index = find_key_region(task);
        // Assign points so they end up wherever the
        // biggest chunk of matrix is for that point
        switch (key_index) {
          case 0: return first_sharding_id + NUMPY_SHARD_TILE_3D_2D_XZ;
          case 1: return first_sharding_id + NUMPY_SHARD_TILE_3D_2D_XY;
          case 2: return first_sharding_id + NUMPY_SHARD_TILE_3D_2D_YZ;
          default: LEGATE_ABORT
        }
      }
      default: LEGATE_ABORT
    }
  } else if (task.tag != 0) {
    // If we've already been perscribed a sharding function then use it
    return task.tag;
  } else {
    // For all other tasks we do the normal tile sharding
    switch (launch_dim) {
      case 1: return first_sharding_id + NUMPY_SHARD_TILE_1D;
      case 2: return first_sharding_id + NUMPY_SHARD_TILE_2D;
      case 3: return first_sharding_id + NUMPY_SHARD_TILE_3D;
      default: LEGATE_ABORT
    }
  }
  // Keep the compiler happy
  return 0;
}

//--------------------------------------------------------------------------
NumPyShardingFunctor* NumPyMapper::find_sharding_functor(ShardingID sid)
//--------------------------------------------------------------------------
{
  assert(first_sharding_id <= sid);
  assert(sid < (first_sharding_id + NUMPY_SHARD_LAST));
  return NumPyShardingFunctor::sharding_functors[sid - first_sharding_id];
}

//--------------------------------------------------------------------------
void NumPyMapper::map_inline(const MapperContext ctx,
                             const InlineMapping& inline_op,
                             const MapInlineInput& input,
                             MapInlineOutput& output)
//--------------------------------------------------------------------------
{
  const std::vector<PhysicalInstance>& valid = input.valid_instances;
  const RegionRequirement& req               = inline_op.requirement;
  output.chosen_instances.resize(req.privilege_fields.size());
  unsigned index = 0;
  std::vector<PhysicalInstance> needed_acquires;
  for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
       it != req.privilege_fields.end();
       it++, index++)
    if (map_numpy_array(ctx,
                        inline_op,
                        0,
                        req.region,
                        *it,
                        local_system_memory,
                        inline_op.parent_task->current_proc,
                        valid,
                        output.chosen_instances[index],
                        false /*memoize*/,
                        req.redop))
      needed_acquires.push_back(output.chosen_instances[index]);
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    std::set<PhysicalInstance> failed_instances;
    filter_failed_acquires(needed_acquires, failed_instances);
    // Now go through all the fields for the instances and try and remap
    std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
    for (unsigned idx = 0; idx < output.chosen_instances.size(); idx++, fit++) {
      if (failed_instances.find(output.chosen_instances[idx]) == failed_instances.end()) continue;
      // Now try to remap it
      if (map_numpy_array(ctx,
                          inline_op,
                          0 /*idx*/,
                          req.region,
                          *fit,
                          local_system_memory,
                          inline_op.parent_task->current_proc,
                          valid,
                          output.chosen_instances[idx],
                          false /*memoize*/))
        needed_acquires.push_back(output.chosen_instances[idx]);
    }
  }
}

//--------------------------------------------------------------------------
void NumPyMapper::select_inline_sources(const MapperContext ctx,
                                        const InlineMapping& inline_op,
                                        const SelectInlineSrcInput& input,
                                        SelectInlineSrcOutput& output)
//--------------------------------------------------------------------------
{
  numpy_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const InlineMapping& inline_op,
                                   const InlineProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling yet for inline mappings
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::map_copy(const MapperContext ctx,
                           const Copy& copy,
                           const MapCopyInput& input,
                           MapCopyOutput& output)
//--------------------------------------------------------------------------
{
  // We should always be able to materialize instances of the things
  // we are copying so make concrete source instances
  std::vector<PhysicalInstance> needed_acquires;
  Memory target_memory = local_system_memory;
  if (copy.is_index_space) {
    // If we've got GPUs, assume we're using them
    if (!local_gpus.empty() || !local_omps.empty()) {
      const ShardingID sid          = select_sharding_functor(copy);
      NumPyShardingFunctor* functor = find_sharding_functor(sid);
      const unsigned local_index =
        functor->localize(copy.index_point, copy.index_domain, total_nodes, local_node);
      if (!local_gpus.empty()) {
        const Processor proc = local_gpus[local_index % local_gpus.size()];
        target_memory        = local_frame_buffers[proc];
      } else {
        const Processor proc = local_omps[local_index % local_omps.size()];
        target_memory        = local_numa_domains[proc];
      }
    }
  } else {
    // If we have just one local GPU then let's use it, otherwise punt to CPU
    // since it's not clear which one we should use
    if (local_frame_buffers.size() == 1) target_memory = local_frame_buffers.begin()->second;
  }
  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
    const RegionRequirement& src_req = copy.src_requirements[idx];
    output.src_instances[idx].resize(src_req.privilege_fields.size());
    const std::vector<PhysicalInstance>& src_valid = input.src_instances[idx];
    unsigned fidx                                  = 0;
    const bool memoize_src                         = ((src_req.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
    for (std::set<FieldID>::const_iterator it = src_req.privilege_fields.begin();
         it != src_req.privilege_fields.end();
         it++) {
      if (find_existing_instance(
            src_req.region, *it, target_memory, output.src_instances[idx][fidx]) ||
          map_numpy_array(ctx,
                          copy,
                          idx,
                          src_req.region,
                          *it,
                          target_memory,
                          Processor::NO_PROC,
                          src_valid,
                          output.src_instances[idx][fidx],
                          memoize_src))
        needed_acquires.push_back(output.src_instances[idx][fidx]);
    }
    const RegionRequirement& dst_req = copy.dst_requirements[idx];
    output.dst_instances[idx].resize(dst_req.privilege_fields.size());
    const std::vector<PhysicalInstance>& dst_valid = input.dst_instances[idx];
    fidx                                           = 0;
    const bool memoize_dst =
      ((dst_req.tag & NUMPY_NO_MEMOIZE_TAG) == 0) && (dst_req.privilege != LEGION_REDUCE);
    for (std::set<FieldID>::const_iterator it = dst_req.privilege_fields.begin();
         it != dst_req.privilege_fields.end();
         it++) {
      if (((dst_req.redop == 0) &&
           find_existing_instance(
             dst_req.region, *it, target_memory, output.dst_instances[idx][fidx])) ||
          map_numpy_array(ctx,
                          copy,
                          copy.src_requirements.size() + idx,
                          dst_req.region,
                          *it,
                          target_memory,
                          Processor::NO_PROC,
                          dst_valid,
                          output.dst_instances[idx][fidx],
                          memoize_dst,
                          dst_req.redop))
        needed_acquires.push_back(output.dst_instances[idx][fidx]);
    }
    if (idx < copy.src_indirect_requirements.size()) {
      const RegionRequirement& src_idx = copy.src_indirect_requirements[idx];
      assert(src_idx.privilege_fields.size() == 1);
      const FieldID fid                              = *(src_idx.privilege_fields.begin());
      const std::vector<PhysicalInstance>& idx_valid = input.src_indirect_instances[idx];
      const bool memoize_idx                         = ((src_idx.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
      if (find_existing_instance(
            src_idx.region, fid, target_memory, output.src_indirect_instances[idx]) ||
          map_numpy_array(ctx,
                          copy,
                          idx,
                          src_idx.region,
                          fid,
                          target_memory,
                          Processor::NO_PROC,
                          idx_valid,
                          output.src_indirect_instances[idx],
                          memoize_idx))
        needed_acquires.push_back(output.src_indirect_instances[idx]);
    }
    if (idx < copy.dst_indirect_requirements.size()) {
      const RegionRequirement& dst_idx = copy.dst_indirect_requirements[idx];
      assert(dst_idx.privilege_fields.size() == 1);
      const FieldID fid                              = *(dst_idx.privilege_fields.begin());
      const std::vector<PhysicalInstance>& idx_valid = input.dst_indirect_instances[idx];
      const bool memoize_idx                         = ((dst_idx.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
      if (find_existing_instance(
            dst_idx.region, fid, target_memory, output.dst_indirect_instances[idx]) ||
          map_numpy_array(ctx,
                          copy,
                          idx,
                          dst_idx.region,
                          fid,
                          target_memory,
                          Processor::NO_PROC,
                          idx_valid,
                          output.dst_indirect_instances[idx],
                          memoize_idx))
        needed_acquires.push_back(output.dst_indirect_instances[idx]);
    }
  }
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    // If we failed to acquire any of the instances we need to prune them
    // out of the mapper's data structure so do that first
    std::set<PhysicalInstance> failed_acquires;
    filter_failed_acquires(needed_acquires, failed_acquires);
    // Now go through and try to remap region requirements with failed acquisitions
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
      const RegionRequirement& src_req               = copy.src_requirements[idx];
      const std::vector<PhysicalInstance>& src_valid = input.src_instances[idx];
      unsigned fidx                                  = 0;
      const bool memoize_src                         = ((src_req.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
      for (std::set<FieldID>::const_iterator it = src_req.privilege_fields.begin();
           it != src_req.privilege_fields.end();
           it++) {
        if (failed_acquires.find(output.src_instances[idx][fidx]) == failed_acquires.end())
          continue;
        if (map_numpy_array(ctx,
                            copy,
                            idx,
                            src_req.region,
                            *it,
                            target_memory,
                            Processor::NO_PROC,
                            src_valid,
                            output.src_instances[idx][fidx],
                            memoize_src))
          needed_acquires.push_back(output.src_instances[idx][fidx]);
      }
      const RegionRequirement& dst_req = copy.dst_requirements[idx];
      output.dst_instances[idx].resize(dst_req.privilege_fields.size());
      const std::vector<PhysicalInstance>& dst_valid = input.dst_instances[idx];
      fidx                                           = 0;
      const bool memoize_dst =
        ((dst_req.tag & NUMPY_NO_MEMOIZE_TAG) == 0) && (dst_req.privilege != LEGION_REDUCE);
      for (std::set<FieldID>::const_iterator it = dst_req.privilege_fields.begin();
           it != dst_req.privilege_fields.end();
           it++) {
        if (failed_acquires.find(output.dst_instances[idx][fidx]) == failed_acquires.end())
          continue;
        if (map_numpy_array(ctx,
                            copy,
                            copy.src_requirements.size() + idx,
                            dst_req.region,
                            *it,
                            target_memory,
                            Processor::NO_PROC,
                            dst_valid,
                            output.dst_instances[idx][fidx],
                            memoize_dst,
                            dst_req.redop))
          needed_acquires.push_back(output.dst_instances[idx][fidx]);
      }
      if (idx < copy.src_indirect_requirements.size()) {
        const RegionRequirement& src_idx = copy.src_indirect_requirements[idx];
        assert(src_idx.privilege_fields.size() == 1);
        const FieldID fid                              = *(src_idx.privilege_fields.begin());
        const std::vector<PhysicalInstance>& idx_valid = input.src_indirect_instances[idx];
        const bool memoize_idx = ((src_idx.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
        if ((failed_acquires.find(output.src_indirect_instances[idx]) != failed_acquires.end()) &&
            map_numpy_array(ctx,
                            copy,
                            idx,
                            src_idx.region,
                            fid,
                            target_memory,
                            Processor::NO_PROC,
                            idx_valid,
                            output.src_indirect_instances[idx],
                            memoize_idx))
          needed_acquires.push_back(output.src_indirect_instances[idx]);
      }
      if (idx < copy.dst_indirect_requirements.size()) {
        const RegionRequirement& dst_idx = copy.dst_indirect_requirements[idx];
        assert(dst_idx.privilege_fields.size() == 1);
        const FieldID fid                              = *(dst_idx.privilege_fields.begin());
        const std::vector<PhysicalInstance>& idx_valid = input.dst_indirect_instances[idx];
        const bool memoize_idx = ((dst_idx.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
        if ((failed_acquires.find(output.dst_indirect_instances[idx]) != failed_acquires.end()) &&
            map_numpy_array(ctx,
                            copy,
                            idx,
                            dst_idx.region,
                            fid,
                            target_memory,
                            Processor::NO_PROC,
                            idx_valid,
                            output.dst_indirect_instances[idx],
                            memoize_idx))
          needed_acquires.push_back(output.dst_indirect_instances[idx]);
      }
    }
  }
}

//--------------------------------------------------------------------------
void NumPyMapper::select_copy_sources(const MapperContext ctx,
                                      const Copy& copy,
                                      const SelectCopySrcInput& input,
                                      SelectCopySrcOutput& output)
//--------------------------------------------------------------------------
{
  numpy_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void NumPyMapper::speculate(const MapperContext ctx, const Copy& copy, SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const Copy& copy,
                                   const CopyProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling for copies yet
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Copy& copy,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = select_sharding_functor(copy);
}

//--------------------------------------------------------------------------
ShardingID NumPyMapper::select_sharding_functor(const Copy& copy)
//--------------------------------------------------------------------------
{
  // If we've already been prescribed a sharding functor then use it
  if (copy.tag != 0) return copy.tag;
  // This is easy for copies, we just use the index space domain for now
  switch (copy.index_domain.get_dim()) {
    case 1: return first_sharding_id + NUMPY_SHARD_TILE_1D;
    case 2: return first_sharding_id + NUMPY_SHARD_TILE_2D;
    case 3: return first_sharding_id + NUMPY_SHARD_TILE_3D;
    default: LEGATE_ABORT
  }
  // Keep the compiler happy
  return 0;
}

//--------------------------------------------------------------------------
void NumPyMapper::map_close(const MapperContext ctx,
                            const Close& close,
                            const MapCloseInput& input,
                            MapCloseOutput& output)
//--------------------------------------------------------------------------
{
  // Map everything with composite instances for now
  output.chosen_instances.push_back(PhysicalInstance::get_virtual_instance());
}

//--------------------------------------------------------------------------
void NumPyMapper::select_close_sources(const MapperContext ctx,
                                       const Close& close,
                                       const SelectCloseSrcInput& input,
                                       SelectCloseSrcOutput& output)
//--------------------------------------------------------------------------
{
  numpy_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const Close& close,
                                   const CloseProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling yet for legate
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Close& close,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::map_acquire(const MapperContext ctx,
                              const Acquire& acquire,
                              const MapAcquireInput& input,
                              MapAcquireOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do
}

//--------------------------------------------------------------------------
void NumPyMapper::speculate(const MapperContext ctx,
                            const Acquire& acquire,
                            SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const Acquire& acquire,
                                   const AcquireProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling for legate yet
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Acquire& acquire,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::map_release(const MapperContext ctx,
                              const Release& release,
                              const MapReleaseInput& input,
                              MapReleaseOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do
}

//--------------------------------------------------------------------------
void NumPyMapper::select_release_sources(const MapperContext ctx,
                                         const Release& release,
                                         const SelectReleaseSrcInput& input,
                                         SelectReleaseSrcOutput& output)
//--------------------------------------------------------------------------
{
  numpy_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void NumPyMapper::speculate(const MapperContext ctx,
                            const Release& release,
                            SpeculativeOutput& output)
//--------------------------------------------------------------------------
{
  output.speculate = false;
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const Release& release,
                                   const ReleaseProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling for legate yet
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Release& release,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_partition_projection(const MapperContext ctx,
                                              const Partition& partition,
                                              const SelectPartitionProjectionInput& input,
                                              SelectPartitionProjectionOutput& output)
//--------------------------------------------------------------------------
{
  // If we have an open complete partition then use it
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions[0];
  else
    output.chosen_partition = LogicalPartition::NO_PART;
}

//--------------------------------------------------------------------------
void NumPyMapper::map_partition(const MapperContext ctx,
                                const Partition& partition,
                                const MapPartitionInput& input,
                                MapPartitionOutput& output)
//--------------------------------------------------------------------------
{
  const RegionRequirement& req = partition.requirement;
  output.chosen_instances.resize(req.privilege_fields.size());
  const std::vector<PhysicalInstance>& valid = input.valid_instances;
  std::vector<PhysicalInstance> needed_acquires;
  unsigned fidx      = 0;
  const bool memoize = ((req.tag & NUMPY_NO_MEMOIZE_TAG) == 0);
  for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
       it != req.privilege_fields.end();
       it++) {
    if (find_existing_instance(req.region,
                               *it,
                               local_system_memory,
                               output.chosen_instances[fidx],
                               Strictness::strict) ||
        map_numpy_array(ctx,
                        partition,
                        0,
                        req.region,
                        *it,
                        local_system_memory,
                        Processor::NO_PROC,
                        valid,
                        output.chosen_instances[fidx],
                        memoize)) {
      needed_acquires.push_back(output.chosen_instances[fidx]);
    }
  }
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    std::set<PhysicalInstance> failed_instances;
    filter_failed_acquires(needed_acquires, failed_instances);
    // Now go through all the fields for the instances and try and remap
    std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
    for (unsigned idx = 0; idx < output.chosen_instances.size(); idx++, fit++) {
      if (failed_instances.find(output.chosen_instances[idx]) == failed_instances.end()) continue;
      // Now try to remap it
      if (map_numpy_array(ctx,
                          partition,
                          0 /*idx*/,
                          req.region,
                          *fit,
                          local_system_memory,
                          Processor::NO_PROC,
                          valid,
                          output.chosen_instances[idx],
                          memoize))
        needed_acquires.push_back(output.chosen_instances[idx]);
    }
  }
}

//--------------------------------------------------------------------------
void NumPyMapper::select_partition_sources(const MapperContext ctx,
                                           const Partition& partition,
                                           const SelectPartitionSrcInput& input,
                                           SelectPartitionSrcOutput& output)
//--------------------------------------------------------------------------
{
  numpy_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

//--------------------------------------------------------------------------
void NumPyMapper::report_profiling(const MapperContext ctx,
                                   const Partition& partition,
                                   const PartitionProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // No profiling yet
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Partition& partition,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = select_sharding_functor(partition);
}

//--------------------------------------------------------------------------
ShardingID NumPyMapper::select_sharding_functor(const Partition& partition)
//--------------------------------------------------------------------------
{
  // This is easy for partitions, we just use the index space domain for now
  switch (partition.index_domain.get_dim()) {
    case 1: return first_sharding_id + NUMPY_SHARD_TILE_1D;
    case 2: return first_sharding_id + NUMPY_SHARD_TILE_2D;
    case 3: return first_sharding_id + NUMPY_SHARD_TILE_3D;
    default: LEGATE_ABORT
  }
  // Keep the compiler happy
  return 0;
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const Fill& fill,
                                          const SelectShardingFunctorInput& input,
                                          SelectShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  output.chosen_functor = select_sharding_functor(fill);
}

//--------------------------------------------------------------------------
ShardingID NumPyMapper::select_sharding_functor(const Fill& fill)
//--------------------------------------------------------------------------
{
  // If we've already been perscribed a sharding functor then use it
  if (fill.tag != 0) return fill.tag;
  // This is easy for fills, we just use the index space domain for now
  switch (fill.index_domain.get_dim()) {
    case 1: return first_sharding_id + NUMPY_SHARD_TILE_1D;
    case 2: return first_sharding_id + NUMPY_SHARD_TILE_2D;
    case 3: return first_sharding_id + NUMPY_SHARD_TILE_3D;
    default: LEGATE_ABORT
  }
  // Keep the compiler happy
  return 0;
}

//--------------------------------------------------------------------------
void NumPyMapper::configure_context(const MapperContext ctx,
                                    const Task& task,
                                    ContextConfigOutput& output)
//--------------------------------------------------------------------------
{
  // Use the defaults currently
}

//--------------------------------------------------------------------------
void NumPyMapper::pack_tunable(const int value, Mapper::SelectTunableOutput& output)
//--------------------------------------------------------------------------
{
  int* result  = (int*)malloc(sizeof(value));
  *result      = value;
  output.value = result;
  output.size  = sizeof(value);
}

//--------------------------------------------------------------------------
void NumPyMapper::select_tunable_value(const MapperContext ctx,
                                       const Task& task,
                                       const SelectTunableInput& input,
                                       SelectTunableOutput& output)
//--------------------------------------------------------------------------
{
  switch (input.tunable_id) {
    case NUMPY_TUNABLE_NUM_PIECES: {
      if (!local_gpus.empty()) {  // If we have GPUs, use those
        pack_tunable(local_gpus.size() * total_nodes, output);
      } else if (!local_omps.empty()) {  // Otherwise use OpenMP procs
        pack_tunable(local_omps.size() * total_nodes, output);
      } else {  // Otherwise use the CPUs
        pack_tunable(local_cpus.size() * total_nodes, output);
      }
      break;
    }
    case NUMPY_TUNABLE_NUM_GPUS: {
      if (!local_gpus.empty())
        pack_tunable(local_gpus.size() * total_nodes, output);  // assume symmetry
      else
        pack_tunable(0, output);
      break;
    }
    case NUMPY_TUNABLE_TOTAL_NODES: {
      pack_tunable(total_nodes, output);
      break;
    }
    case NUMPY_TUNABLE_LOCAL_CPUS: {
      pack_tunable(local_cpus.size(), output);
      break;
    }
    case NUMPY_TUNABLE_LOCAL_GPUS: {
      pack_tunable(local_gpus.size(), output);
      break;
    }
    case NUMPY_TUNABLE_LOCAL_OPENMPS: {
      pack_tunable(local_omps.size(), output);
      break;
    }
    case NUMPY_TUNABLE_MIN_SHARD_VOLUME: {
      // TODO: make these profile guided
      if (!local_gpus.empty())
        // Make sure we can get at least 1M elements on each GPU
        pack_tunable(min_gpu_chunk, output);
      else if (!local_omps.empty())
        // Make sure we get at least 128K elements on each OpenMP
        pack_tunable(min_omp_chunk, output);
      else
        // Make sure we can get at least 8KB elements on each CPU
        pack_tunable(min_cpu_chunk, output);
      break;
    }
    case NUMPY_TUNABLE_MAX_EAGER_VOLUME: {
      // TODO: make these profile guided
      if (eager_fraction > 0) {
        if (!local_gpus.empty())
          pack_tunable(min_gpu_chunk / eager_fraction, output);
        else if (!local_omps.empty())
          pack_tunable(min_omp_chunk / eager_fraction, output);
        else
          pack_tunable(min_cpu_chunk / eager_fraction, output);
      } else
        pack_tunable(0, output);
      break;
    }
    case NUMPY_TUNABLE_FIELD_REUSE_SIZE: {
      // We assume that all memories of the same kind are symmetric in size
      size_t local_mem_size;
      if (!local_gpus.empty()) {
        assert(!local_frame_buffers.empty());
        local_mem_size = local_frame_buffers.begin()->second.capacity();
        local_mem_size *= local_frame_buffers.size();
      } else if (!local_omps.empty()) {
        assert(!local_numa_domains.empty());
        local_mem_size = local_numa_domains.begin()->second.capacity();
        local_mem_size *= local_numa_domains.size();
      } else
        local_mem_size = local_system_memory.capacity();
      // Multiply this by the total number of nodes and then scale by the frac
      const size_t global_mem_size  = local_mem_size * total_nodes;
      const size_t field_reuse_size = global_mem_size / field_reuse_frac;
      // Pack this one explicity since it must be of size 8
      size_t* result = (size_t*)malloc(sizeof(field_reuse_size));
      *result        = field_reuse_size;
      output.value   = result;
      output.size    = sizeof(field_reuse_size);
      break;
    }
    case NUMPY_TUNABLE_FIELD_REUSE_FREQUENCY: {
      pack_tunable(field_reuse_freq, output);
      break;
    }
    default: LEGATE_ABORT  // unknown tunable value
  }
}

//--------------------------------------------------------------------------
void NumPyMapper::select_sharding_functor(const MapperContext ctx,
                                          const MustEpoch& epoch,
                                          const SelectShardingFunctorInput& input,
                                          MustEpochShardingFunctorOutput& output)
//--------------------------------------------------------------------------
{
  // No must epoch launches in legate
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::memoize_operation(const MapperContext ctx,
                                    const Mappable& mappable,
                                    const MemoizeInput& input,
                                    MemoizeOutput& output)
//--------------------------------------------------------------------------
{
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::map_must_epoch(const MapperContext ctx,
                                 const MapMustEpochInput& input,
                                 MapMustEpochOutput& output)
//--------------------------------------------------------------------------
{
  // No must epoch launches in legate
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::map_dataflow_graph(const MapperContext ctx,
                                     const MapDataflowGraphInput& input,
                                     MapDataflowGraphOutput& output)
//--------------------------------------------------------------------------
{
  // Not supported yet
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::select_tasks_to_map(const MapperContext ctx,
                                      const SelectMappingInput& input,
                                      SelectMappingOutput& output)
//--------------------------------------------------------------------------
{
  // Just map all the ready tasks
  for (std::list<const Task*>::const_iterator it = input.ready_tasks.begin();
       it != input.ready_tasks.end();
       it++)
    output.map_tasks.insert(*it);
}

//--------------------------------------------------------------------------
void NumPyMapper::select_steal_targets(const MapperContext ctx,
                                       const SelectStealingInput& input,
                                       SelectStealingOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do, no stealing in the leagte mapper currently
}

//--------------------------------------------------------------------------
void NumPyMapper::permit_steal_request(const MapperContext ctx,
                                       const StealRequestInput& input,
                                       StealRequestOutput& output)
//--------------------------------------------------------------------------
{
  // Nothing to do, no stealing in the legate mapper currently
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::handle_message(const MapperContext ctx, const MapperMessage& message)
//--------------------------------------------------------------------------
{
  // We shouldn't be receiving any messages currently
  LEGATE_ABORT
}

//--------------------------------------------------------------------------
void NumPyMapper::handle_task_result(const MapperContext ctx, const MapperTaskResult& result)
  //--------------------------------------------------------------------------
  {// Nothing to do since we should never get one of these
   LEGATE_ABORT}

//--------------------------------------------------------------------------
NumPyOpCode NumPyMapper::decode_task_id(TaskID tid)
//--------------------------------------------------------------------------
{
  // This better be a NumPy task
  assert((first_numpy_task_id <= tid) && (tid <= last_numpy_task_id));
  return static_cast<NumPyOpCode>(tid - first_numpy_task_id);
}

//--------------------------------------------------------------------------
/*static*/ unsigned NumPyMapper::extract_env(const char* env_name,
                                             const unsigned default_value,
                                             const unsigned test_value)
//--------------------------------------------------------------------------
{
  const char* legate_test = getenv("NUMPY_TEST");
  if (legate_test != NULL) return test_value;
  const char* env_value = getenv(env_name);
  if (env_value == NULL)
    return default_value;
  else
    return atoi(env_value);
}

}  // namespace numpy
}  // namespace legate
