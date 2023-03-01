# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.errors import NoUri
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.logging import get_node_location, getLogger

from . import PARALLEL_SAFE, SphinxParallelSpec

log = getLogger(__name__)

SKIP = (
    "cunumeric.cast",
    "cunumeric.ndarray.__array_function__",
    "cunumeric.ndarray.__array_ufunc__",
    "cunumeric.ndarray.__format__",
    "cunumeric.ndarray.__hash__",
    "cunumeric.ndarray.__iter__",
    "cunumeric.ndarray.__radd__",
    "cunumeric.ndarray.__rand__",
    "cunumeric.ndarray.__rdivmod__",
    "cunumeric.ndarray.__reduce_ex__",
    "cunumeric.ndarray.__rfloordiv__",
    "cunumeric.ndarray.__rmod__",
    "cunumeric.ndarray.__rmul__",
    "cunumeric.ndarray.__ror__",
    "cunumeric.ndarray.__rpow__",
    "cunumeric.ndarray.__rsub__",
    "cunumeric.ndarray.__rtruediv__",
    "cunumeric.ndarray.__rxor__",
    "cunumeric.ndarray.__sizeof__",
)

MISSING: list[tuple[str, str]] = []


class MissingRefs(SphinxPostTransform):
    default_priority = 5

    def run(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.pending_xref):
            self._check_target(node)

    def _check_target(self, node: Any) -> None:
        target = node["reftarget"]

        if not target.startswith("cunumeric.") or target in SKIP:
            return

        domain = self.env.domains[node["refdomain"]]

        assert self.app.builder is not None

        try:
            uri = domain.resolve_xref(
                self.env,
                node.get("refdoc", self.env.docname),
                self.app.builder,
                node["reftype"],
                target,
                node,
                nodes.TextElement("", ""),
            )
        except NoUri:
            uri = None

        if uri is None:
            loc = get_node_location(node)
            log.warning(
                f"Cunumeric reference missing a target: {loc}: {target}",
                type="ref",
            )


def setup(app: Sphinx) -> SphinxParallelSpec:
    app.add_post_transform(MissingRefs)
    return PARALLEL_SAFE
