// Some of the constants are read in from load_scale_json
var constants = {
  margin_left: 350,
  margin_right: 75,
  margin_bottom: 50,
  margin_top: 50,
  min_feature_width: 3,
  min_gap_width: 1,
  util_height: 100,
  util_levels: 4,
  elem_separation: 2,
}

var op_dependencies = {};
var base_map = {};
var prof_uid_map = {};

// Contains the children for each util file
var util_files = {};

var state = {};
var mouseX = 0;
var utilline = d3.svg.line()
    .interpolate("step-after")
    .x(function(d) { return state.x(+d.time); })
    .y(function(d) { return state.y(+d.count); });
var timeBisector = d3.bisector(function(d) { return d.time; }).left;
// const num_colors = 20;
// var color = d3.scale.linear()
//   .domain([0, num_colors])
//   .range(["green", "blue", "red", "yellow"]);

String.prototype.hashCode = function() {
  var hash = 0;
  if (this.length == 0) return hash;
  for (i = 0; i < this.length; i++) {
    var c = this.charCodeAt(i);
    var b = this.charCodeAt(c % this.length);
    hash = ((hash<<5)-hash)^c^b;
    hash = hash & hash; 
  }
  return Math.abs(hash);
}

function parseURLParameters() {
  var match,
  pl     = /\+/g,  // Regex for replacing addition symbol with a space
  search = /([^&=]+)=?([^&]*)/g,
  decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
  query  = window.location.search.substring(1);

  var urlParams = {};
  while (match = search.exec(query))
    urlParams[decode(match[1])] = decode(match[2]);

  if ("collapseAll" in urlParams)
    state.collapseAll = (urlParams["collapseAll"].toLowerCase() === "true");

  if ("resolution" in urlParams)
    state.resolution = Math.max(1, parseFloat(urlParams["resolution"]));
  // adjust zoom
  var zstart = constants.start;
  if ("start" in urlParams)
    zstart = Math.max(constants.start, parseFloat(urlParams["start"]));

  var zend = constants.end;
  if ("end" in urlParams)
    zend = Math.min(constants.end, parseFloat(urlParams["end"]));

  if(zstart < zend) {
    // set zoom to get: (zend - start) * zoom * scale = $("#timeline").width()
    var newZoom = $("#timeline").width() / state.scale / (zend - zstart);
    adjustZoom(newZoom, false);

    // set scrollLeft to get:  zstart * zoom * state.scale = scrollLeft()
    $("#timeline").scrollLeft(convertToPos(state, zstart));
  }

  if ("search" in urlParams) {
    state.searchEnabled = true;
    searchRegex = new Array(sizeHistory);
    currentPos = 0;
    nextPos = 1;
    searchRegex[currentPos] = new RegExp(urlParams["search"]);
  }
}

function makeTimelineTransparent() {
  state.timelineSvg.select("g#timeline").style("opacity", "0.1");
  $("#timeline").css("overflow-x", "hidden");
  state.timelineSvg.select("g#lines").style("opacity", "0.1");
  state.timelineSvg.select("g.locator").style("opacity", "0.1");
  state.timelineSvg.selectAll("path.util").style("opacity", "0.1");
}

function makeTimelineOpaque() {
  state.timelineSvg.select("g#timeline").style("opacity", "1.0");
  $("#timeline").css("overflow-x", "scroll");
  state.timelineSvg.select("g#lines").style("opacity", "1.0");
  state.timelineSvg.select("g.locator").style("opacity", "1.0");
  state.timelineSvg.selectAll("path.util").style("opacity", "1.0");
}

function mouseMoveHandlerWhenDown() {
  var p = d3.mouse(this);
  var select_block = state.timelineSvg.select("rect.select-block");
  var select_text = state.timelineSvg.select("text.select-block");
  var newWidth = Math.abs(p[0] - mouseX);
  select_block.attr("width", newWidth);
  if (p[0] >= mouseX) {
    select_block.attr("x", mouseX);
    select_text.attr("x", mouseX + (p[0] - mouseX) / 2);
  } else {
    select_block.attr("x", p[0]);
    select_text.attr("x", p[0] + (mouseX - p[0]) / 2);
  }
  var time = convertToTime(state, newWidth);
  select_text.text(getTimeString(time, time));
}

function mouseMoveHandlerWhenUp() {
  var p = d3.mouse(this);
  var x = parseFloat(p[0]);
  var scrollLeft = $("#timeline").scrollLeft();
  var paneWidth = $("#timeline").width();
  var currentTime = convertToTime(state, x);

  state.timelineSvg.select("g.locator").remove();
  var locator = state.timelineSvg.append("g").attr("class", "locator");
  locator.insert("line")
    .attr({
      x1: p[0],
      y1: 0,
      x2: p[0],
      y2: p[1] - state.thickness / 2,
      class: "locator",
    });
  locator.append("line")
    .attr({
      x1: p[0],
      y1: p[1] + state.thickness / 2,
      x2: p[0],
      y2: state.height,
      class: "locator",
    });
  var locatorText = locator.append("text");
  var text = getTimeString(currentTime, convertToTime(state, paneWidth));
  locatorText.attr("class", "locator").text(text)
  if ((x - scrollLeft) < paneWidth - 100) {
    locatorText.attr({x: x + 2, y: $(window).scrollTop() + 10});
    locatorText.attr("anchor", "start");
  }
  else {
    locatorText.attr({x: x - 2 - text.length * 7, y: $(window).scrollTop() + 10});
    locatorText.attr("anchor", "end");
  }
}

function mouseDownHandler() {
  state.timelineSvg.select("g.locator").remove();
  var p = d3.mouse(this);
  state.timelineSvg.append("rect")
    .attr({
      x : p[0],
      y : 0,
      class : "select-block",
      width : 0,
      height : state.height
    });
  state.timelineSvg.append("text")
    .attr({
      x : p[0],
      y : p[1],
      class : "select-block",
      anchor : "middle",
      "text-anchor" : "middle",
    }).text("0 us");
  mouseX = p[0];
  state.timelineSvg.on("mousemove", null);
  state.timelineSvg.on("mousemove", mouseMoveHandlerWhenDown);
  $(document).off("keydown");
}

function get_kind(name) {
  var util_regex = /\((.*?)\)/;
  var kind_match = util_regex.exec(name);
  if (kind_match) {
    return kind_match[1];
  }
}

function getLineColor(elem) {
  var kind = get_kind(elem.text);
  const colorMap = {
    "CPU": "steelblue",
    "GPU": "olivedrab",
    "Utility": "crimson",
    "IO": "orangered",
    "Proc Group": "orangered",
    "Proc Set": "orangered",
    "OpenMP": "orangered",
    "Python": "olivedrab",
    "System Memory": "olivedrab",
    "GASNet Global Memory": "crimson",
    "Registered Memory": "darkmagenta",
    "Socket Memory": "orangered",
    "Zero-Copy Memory": "crimson",
    "Framebuffer Memory": "blue",
    "Disk Memory": "darkgoldenrod",
    "HDF5 Memory": "olivedrab",
    "File Memory": "orangered",
    "L3 Cache Memory": "crimson",
    "L2 Cache Memory": "darkmagenta",
    "L1 Cache Memory": "olivedrab",
    "Channel": "orangered"
  };
  return colorMap[kind];
}

function drawUtil() {
  // TODO: Add to state
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  var start_time = convertToTime(state, windowStart);
  var end_time = convertToTime(state, windowEnd);
  var filteredUtilData = [];
  for (var i = 0; i < state.flattenedLayoutData.length; i++) {
    var elem = state.flattenedLayoutData[i];
    if (elem.type == "util" && elem.enabled && elem.loaded && elem.visible) {
      filteredUtilData.push(filterUtilData(elem));
    }
  }

  state.x = d3.scale.linear().range([0, convertToPos(state, end_time)]);
  state.x.domain([0, end_time]);
  state.timelineSvg.selectAll("rect.util").remove();
  state.timelineSvg.selectAll("path.util").remove();
  var paths = state.timelineSvg.selectAll("path")
    .data(filteredUtilData);
  var totalWidth = windowStart + $("#timeline").width();
  paths.enter().append("rect")
    .attr("class", "util")
    .attr("base_y", lineLevelCalculator)
    .attr("y", lineLevelCalculator)
    .attr("x", 0)
    .attr("fill", "transparent")
    .attr("width", totalWidth)
    .attr("height", constants.util_levels * state.thickness)
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseout", function() {
      state.timelineSvg.select("g.focus").remove();
      state.timelineSvg.select("g.utilDesc").remove();
    });
  paths.enter().append("path")
    .attr("base_y", lineLevelCalculator)
    .attr("class", "util")
    .attr("id", function(d, i) { return "util" + i})
    .attr("d", function (d) { return utilline(d.data); })
    .attr("stroke", getLineColor)
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseout", function() {
      state.timelineSvg.select("g.focus").remove();
      state.timelineSvg.select("g.utilDesc").remove();
    })
    .attr("transform",
      function(d) {
        var y = lineLevelCalculator(d);
        return "translate(0," + y + ")"
    });
}


function mouseUpHandler() {
  var p = d3.mouse(this);
  var select_block = state.timelineSvg.select("rect.select-block");
  var prevZoom = state.zoom;
  var selectWidth = parseInt(select_block.attr("width"));
  var svgWidth = state.timelineSvg.attr("width");
  if (state.rangeZoom && selectWidth > 10) {
    var x = select_block.attr("x");
    state.zoomHistory.push({zoom: prevZoom, start: $("#timeline").scrollLeft()});
    adjustZoom(svgWidth / selectWidth, false);
    $("#timeline").scrollLeft(x / prevZoom * state.zoom);
  }

  state.timelineSvg.selectAll("rect.select-block").remove();
  state.timelineSvg.selectAll("text.select-block").remove();
  mouseX = 0;
  state.timelineSvg.on("mousemove", null);
  state.timelineSvg.on("mousemove", mouseMoveHandlerWhenUp);
  $(document).on("keydown", defaultKeydown);
}

function turnOffMouseHandlers() {
  state.timelineSvg.on("mousedown", null);
  state.timelineSvg.on("mouseup", null);
  state.timelineSvg.on("mousemove", null);
  state.timelineSvg.on("mousemove", null);
  // prevent right-click menu
  state.timelineSvg.on("contextmenu", function () {
                          d3.event.preventDefault();
                        });
}

function turnOnMouseHandlers() {
  state.timelineSvg.on("mousedown", mouseDownHandler);
  state.timelineSvg.on("mouseup", mouseUpHandler);
  state.timelineSvg.on("mousemove", mouseMoveHandlerWhenUp);
  // prevent right-click menu
  state.timelineSvg.on("contextmenu", function () {
                          d3.event.preventDefault();
                        });
}

function drawLoaderIcon() {
  var loaderGroup = state.loaderSvg.append("g")
    .attr({
        id: "loader-icon",
    });
  loaderGroup.append("path")
    .attr({
      opacity: 0.2,
      stroke: "steelblue",
      fill: "#000",
      d: "M20.201,5.169c-8.254,0-14.946,6.692-14.946,14.946c0,8.255,6.692,14.946,14.946,14.946s14.946-6.691,14.946-14.946C35.146,11.861,28.455,5.169,20.201,5.169z M20.201,31.749c-6.425,0-11.634-5.208-11.634-11.634c0-6.425,5.209-11.634,11.634-11.634c6.425,0,11.633,5.209,11.633,11.634C31.834,26.541,26.626,31.749,20.201,31.749z"
    });
  var path = loaderGroup.append("path")
    .attr({
      stroke: "steelblue",
      fill: "#000",
      d: "M26.013,10.047l1.654-2.866c-2.198-1.272-4.743-2.012-7.466-2.012h0v3.312h0C22.32,8.481,24.301,9.057,26.013,10.047z"
    });
  path.append("animateTransform")
    .attr({
      attributeType: "xml",
      attributeName: "transform",
      type: "rotate",
      from: "0 20 20",
      to: "360 20 20",
      dur: "0.5s",
      repeatCount: "indefinite"
    });
    state.loaderSvg.select("g").attr("visibility", "hidden");
    state.loaderSvg.attr("width", "0px")
                   .attr("height", "0px");
}

function showLoaderIcon() {
  state.numLoading++;
  state.loaderSvg.select("g").attr("visibility", "visible");
  state.loaderSvg.attr("width", "40px")
                 .attr("height", "40px");
}

function hideLoaderIcon() {
  state.numLoading--;
  if (state.numLoading == 0) {
    state.loaderSvg.select("g").attr("visibility", "hidden");
    state.loaderSvg.attr("width", "0px")
                   .attr("height", "0px");
;
  }
}

function get_time_str(time_val, convert)
{
    if (convert) {
	if (time_val >= 1000000) {
	    var val = time_val/1000000;
	    val = parseFloat(val.toFixed(3))
		return val.toString() + " sec";
	}
	if (time_val >= 1000) {
	    var val = time_val/1000;
	    val = parseFloat(val.toFixed(3))
		return val.toString() + " ms";
	}
    }
    return time_val.toString() + " us";
}

function getMouseOver() {
  var paneWidth = $("#timeline").width();
  var left = paneWidth / 3;
  var right = paneWidth * 2 / 3;
  return function(d, i) {
    var p = d3.mouse(this);
    var x = parseFloat(p[0]);
    var y = timelineLevelCalculator(d) - 5;
    var descView = state.timelineSvg.append("g")
                        .attr("id", "desc");
    var text = descView.append("text")
                       .attr("x", x)
                       .attr("y", y)
                       .attr("class", "desc");
    var depElem = prof_uid_map[d.prof_uid][0];
    if ((depElem.in.length != 0) || (depElem.out.length != 0) || 
        (depElem.children.length !== 0) || (depElem.parents.length !==0 )) {
      d3.select(this).style("cursor", "pointer");
    }
    // descTexts is an array of Texts we will store in the desc view
    var descTexts = [];
    var total = d.end - d.start;
    var delay = d.start - d.ready;
    total = parseFloat(total.toFixed(3))
    delay = parseFloat(delay.toFixed(3))
    var initiation = "";
    // Insert texts in reverse order
    if ((d.ready != undefined) && (d.ready != "") && (delay != 0)) {
	descTexts.push("Ready State: " + get_time_str(delay,false));
    }
    descTexts.push("End:   " + get_time_str(d.end, false));
    descTexts.push("Start: " + get_time_str(d.start, false));
    descTexts.push("Total: " + get_time_str(total, true));
    if ((d.initiation != undefined) && d.initiation != "") {
      descTexts.push("Initiator: " + state.operations[d.initiation].desc);
    } 

   // split d.title
    var titles =  d.title.split("$");
      if (titles.length > 0) {
	  for (var i = titles.length-1; i >= 0; --i) {
	      descTexts.push(titles[i]);
	  }
      }
    var title = text.append("tspan")
      .attr("x", x)
      .attr("dy", -12)
      .attr("class", "desc")
      .text(descTexts[0].replace(/ /g, "\u00A0")); // preserve spacing

    for (var i = 1; i < descTexts.length; ++i) {
      var elem = text.append("tspan")
        .attr("x", x)
        .attr("dy", -12)
        .attr("class", "desc")
        .text(descTexts[i].replace(/ /g, "\u00A0")); // preserve spacing
    }

    var bbox = descView.node().getBBox();
    var padding = 2;
    var rect = descView.insert("rect", "text")
        .attr("x", bbox.x - 2*padding)
        .attr("y", bbox.y - padding)
        .attr("width", bbox.width + (padding*4))
        .attr("height", bbox.height + (padding*2))
        .style("fill", "#222")
        .style("opacity", "0.7");

    var bboxRight = bbox.x + bbox.width;
    var timelineRight = $("#timeline").scrollLeft() + $("#timeline").width();

    // If the box moves off the screen, nudge it back
    if (bboxRight > timelineRight) {
      var translation = -(bboxRight - timelineRight + 20);
      descView.attr("transform", "translate(" + translation + ",0)");
    }
  };
}

var sizeHistory = 10;
var currentPos;
var nextPos;
var searchRegex = null;

function showSlider() {
    $('#lowerLimit').text($('#node_slider').slider("values", 0));
    $('#upperLimit').text($('#node_slider').slider("values", 1));
}

function createMenuList1() {
    var memory_kinds = [];
    var num_nodes = state.num_nodes;
    var num_mems = 0;
    var dropdown_options = [
	                    { value: "ready",
                              id: "ready_all",
			      text: "View",
			      count: 1,
			      low_range: 0,
			      high_range: 1
			    },
			    { value: "memories",
			      id: "memories_all",
			      text: "Memories",
			      count: num_mems,
			      low_range: 0,
			      high_range: num_mems -1
			    }
			    ];

    // memories list
    dropdown_options_memories = [];

    // misc items list
    var pkind = {
	id: "rr",
	value: "rdy",
	count: 1,
	low_range: 0,
	high_range: 1,
	text: "Ready State"
      };

    dropdown_options_misc = [];
    dropdown_options_misc.push(pkind);

    // create the remaining entries
    function appendKind(elem) {
	var value = state.processor_kinds[elem];
	if (elem.includes("Memory") && (value.count >= 0))
	    {
		num_mems = num_mems+1;
		dropdown_options_memories.push(value);
	    }
	if ((value.count >= 0) && (value.low_range != -1)) {
	    dropdown_options.push(value);
	}
    }
    Object.keys(state.processor_kinds).forEach(appendKind);

    // main menu
    d3.select("#dropdown").selectAll("optgroup")
	.data(dropdown_options)
	.enter()
	.append("optgroup")
	.attr('label', function(d) { return d.text;})
	.attr("id", function(d) { return d.id;})
	.attr("value", function(d) { return d.value;});

    for (var j=0; j<dropdown_options.length; j++) {
	var num_nodes_options = [];
	var num_vals=dropdown_options[j].count;
	val = dropdown_options[j].low_range;
	if (dropdown_options[j].id == "memories_all")
	    num_nodes_options = dropdown_options_memories;
	else if (dropdown_options[j].id == "ready_all")
	    num_nodes_options = dropdown_options_misc;
	else {
	    for (var i=0; i<num_vals; i++)
		{
		    var el = {
			value: dropdown_options[j].text + val.toString(),
			id_lo: "node_low" + j.toString() +  i,
			id_hi: "node_hi" + j.toString() + i,
			text: val.toString()
		    };
		    val = val+1;
		    num_nodes_options[i] = el;
		}
	}
	d3.select("#" + dropdown_options[j].id).selectAll("option")
	    .data(num_nodes_options)
	    .enter()
	    .append("option")
	    .text(function(d) { return d.text;})
	    .attr("id", function(d) { return d.id_lo; })
	    .attr("value", function(d) { return d.value; });
    }
    var node_range1 = num_nodes-1;
    $("#node_slider").slider({
	        values: [0,node_range1],
		range: true,
		create: showSlider,
		min: 0,
		max: node_range1,
		step: 1,
		});

    $('#dropdown').multiselect({
      maxHeight:200,
      includeSelectAllOption: true,
	buttonText: function(options) {
	    return 'Profiling Options';
	}
    });
}

function flattenLayoutProfileView() {
    state.flattenedLayoutData = [];
    function appendElem(elem) {
	if (elem.selected) {
	    state.flattenedLayoutData.push(elem);
	    elem.children.forEach(appendElem);
	}
    }
    state.layoutData.forEach(appendElem);
}

function readyStateUpdate() {
    state.flattenedLayoutData = [];
    function appendElem(elem) {
	if (elem.selected) {
	    state.flattenedLayoutData.push(elem);
	    elem.children.forEach(appendElem);
	}
    }
    if (state.ready_selected) {
	state.layoutData.forEach(appendElem);
    }
}

function printSelected() {
    function printElem(elem) {
	if (elem.selected)
	    console.log("Element Selected:");
	else
	    console.log("Element Not Selected:");
	console.log(elem.text, elem.full_text);
	elem.children.forEach(printElem);
    }
    state.layoutData.forEach(printElem);
}


function countSelected() {
    function count(elem) {
	var count = 0;
	function countChildren(child_elem) {
	if (child_elem.selected)
	    count = count+1;
	}
	elem.children.forEach(countChildren);
	if ((count == 0) && (elem.children.length != 0))
	    elem.selected = false;
	if (elem.children.length == 0)
	    elem.selected = elem.selected;
    }
    state.layoutData.forEach(count);
}

function in_channel_node_id(text) {
    var node_regex = /n(\d+)/;
    var match = node_regex.exec(text);
    var node_id;
    if (match) {
	node_id = parseInt(match[1]);
    }
    return node_id;
}

function out_channel_node_id(text) {
    var node_regex = /to\s+\[n(\d+)/;
    var match = node_regex.exec(text);
    var node_id;
    if (match) {
	node_id = parseInt(match[1]);
    }
    return node_id;
}

function in_channel_node(text) {
    var node_regex = /n(\d+)/;
    var match = node_regex.exec(text);
    var node_id;
    if (match) {
	node_id = "Nodes" + parseInt(match[1]);
    }
    return node_id;
}

function out_channel_node(text) {
    var node_regex = /to\s+\[n(\d+)/;
    var match = node_regex.exec(text);
    var node_id;
    if (match) {
	node_id = "Nodes" + parseInt(match[1]);
    }
    return node_id;
}

function in_channel_proc(text) {
    var node_regex = /^\s+\[n\d+\]\[(\w+)\s+(\d+)/;
    var match = node_regex.exec(text);
    var node_id;
    if (match) {
	node_id = match[1] + parseInt(match[2]);
    }
    return node_id;
}

function out_channel_proc(text) {
    // [n...] to [n...][proc m]
    var node_regex = /to\s+\[n\d+\]\[(\w+)\s+(\d+)/;
    var match = node_regex.exec(text);
    var node_id;
    if (match) {
	node_id = match[1] + parseInt(match[2]);
    }
    return node_id;
}

function  in_channel_mem_processor(full_text) {
    var node_regex = /^(\D+Memory)/;
    var match = node_regex.exec(full_text);
    if (match) {
	return match[1];
    }
    return match;
}

function  out_channel_mem_processor(full_text) {
    var node_regex = /to\s+(\D+Memory)/;
    var match = node_regex.exec(full_text);
    if (match) {
	return match[1];
    }
    return undefined;
}

function all_selected(in_node, out_node, in_proc, out_proc, in_mem, out_mem, map_selected) {
    if (in_node != undefined) {
	if (!(in_node in map_selected))
	    return false;
    }
    if (out_node != undefined) {
	if (!(out_node in map_selected))
	    return false;
    }
    if (in_proc != undefined) {
	if (!(in_proc in map_selected))
	    return false;
    }
    if (out_proc != undefined) {
	if (!(out_proc in map_selected))
	    return false;
    }
    if (in_mem != undefined) {
	if (!(in_mem in map_selected))
	    return false;
    }
    if (out_mem != undefined) {
	if (!(out_mem in map_selected))
	    return false;
    }
    return true;
}

function profileSelect(event) {
    if (event.target.id == "node_slider") {
	$('#lowerLimit').text($('#node_slider').slider("values", 0));
	$('#upperLimit').text($('#node_slider').slider("values", 1));
    }
    if (event.target.id == "submit-order") {
	var map_selected = {};
	var count = 0;
	$('#dropdown option:selected').each(function() {
		map_selected[$(this).val()] = true;
		count = count + 1;
	    });
	if (count == 0){
	    text = "Must select at least one option";
	    alert(text);
	}
	else if ((count == 1) && ("rdy" in map_selected)) {
	    text = "Select additional options with View Ready State"
	    alert(text);
	    }
	else {
	    var lower_range =  $('#node_slider').slider("values", 0);
	    var upper_range =  $('#node_slider').slider("values", 1);
	    for (var i=parseInt(lower_range); i<=parseInt(upper_range); i++) {
		var node_val = "Nodes" + i.toString();
		map_selected[node_val] = true;
	    }
	    updateProfileView(map_selected);
	}
    }
}

function redoProfileView() {
  state.timelineSvg.select("g.dependencies").remove();
  state.timelineSvg.selectAll("g.critical_path_lines").remove();
  calculateBases();
  filterAndMergeBlocks(state);
  constants.max_level = calculateVisibileLevels();
  recalculateHeight();
  drawTimeline();
  drawLayout();
}

function updateProfileView(map_selected) {
    // rebuild FlattenedLayoutData
    // update the layoutData element selected field
    function updateSelect(elem) {
	var selected=true;
	var val = get_node_val(elem.text);
	var node_val;
	if (val != undefined)
	    node_val = "Nodes" + val.toString();
	if ((elem.type == "util") && (val != undefined)) {
	    // nodes
	    if (elem.text.includes("node")) {
		if (!(node_val in map_selected))
		    selected=false;
	    }
	    // memories
	    else if (elem.text.includes("Memory")) {
		var kind = get_kind(elem.text);
		var node_id = get_node_val(elem.text);
		if (!(kind in map_selected) || (!(node_val in map_selected)))
		    selected=false;
	    }
	    // channels
	    else if (elem.text.includes("Channel")) {
		if (!(node_val in map_selected))
		    selected=false;
	    }
	}
	// processors
	else if ((elem.type == "proc") && (elem.parent != undefined)) {
	    var kind = get_kind(elem.parent.text);
	    var node_id = get_node_id(elem.full_text);
	    var node_val = "Nodes" + node_id.toString();
	    if (kind in state.processor_kinds) {
		var r = /\d+/;
		var kind_proc = state.processor_kinds[kind];
		// processors
		if (is_proc_string(elem.full_text)) {
		    var val = parseInt(elem.text.match(r));
		    var proc_val = kind + val.toString();
		    if ((!(proc_val in map_selected)) || (!(node_val in map_selected))) {
			selected = false;
		    }
		}
		// add channels that are relevent to the selected nodes and processors
		if (kind == "Channel") {
		    // all have to be included for us to show this
		    // [node n][proc m] to [node p][proc q]
		    var in_node, in_proc, out_node, out_proc, in_mem, out_mem;
		    in_node = in_channel_node(elem.text);
		    out_node = out_channel_node(elem.text);
		    in_proc = in_channel_proc(elem.text);
		    out_proc = out_channel_proc(elem.text);
		    in_mem = in_channel_mem_processor(elem.full_text);
		    out_mem = out_channel_mem_processor(elem.full_text);
		    if (!all_selected(in_node, out_node, in_proc, out_proc, in_mem, out_mem, map_selected))
			selected=false;
		}
		// memory
		if (kind.includes("Memory") && is_mem_string(elem.full_text)) {
		    if ((!(kind in map_selected)) || (!(node_val in map_selected))) {
			selected=false;
		    }
		}
	    }
	}
	elem.selected = selected;
	elem.children.forEach(updateSelect);
    }
    state.layoutData.forEach(updateSelect);
    // add ready state to timeline
    if ("rdy" in map_selected)
	state.ready_selected=true;
    else
	state.ready_selected=false;
    countSelected();
    //    printSelected();
    flattenLayout();
    redoProfileView();
}

function flattenLayout() {
  state.flattenedLayoutData = [];

  function appendElem(elem) {
      if (elem.selected) {
	  state.flattenedLayoutData.push(elem);
	  elem.children.forEach(appendElem);
      }
  }
  state.layoutData.forEach(appendElem);
}

function getProcessors(util_name) {
  var matched_procs = [];

  // returns processors for a given node
  var util_regex = /(node )?(\d+) \((.+?)\)/;
  var util_match = util_regex.exec(util_name);

  if (util_match) {
    var util_node_id = parseInt(util_match[2], 10);
    var util_type = util_match[3];
    var proc_regex;
    if (util_type.includes("Channel")) {
      proc_regex = /(\D+ Memory) 0x1e([a-fA-f0-9]{4})[a-fA-f0-9]{10}.*(Channel)/;
    }
    else if (util_type.includes("Memory")) {
      proc_regex = /(\D+ Memory) 0x1e([a-fA-f0-9]{4})[a-fA-f0-9]{10}$/;
    } else {
      proc_regex = /(\D+) Processor 0x1d([a-fA-f0-9]{4})/;
    }
    state.processors.forEach(function(proc) {
      var proc_match = proc_regex.exec(proc.full_text);
      if (proc_match) {
        var proc_type = proc_match[1];
        var proc_node_id = parseInt(proc_match[2], 16);
        //  record in and out channels
        if (util_type.includes("Channel")) {
	    proc_type = proc_match[3];
	    var in_node = in_channel_node_id(proc.text);
	    var out_node = out_channel_node_id(proc.text);
	    if (in_node == util_node_id || (out_node == util_node_id) && (proc_type  == util_type))
		matched_procs.push(proc);
	}
	  else if ((proc_node_id == util_node_id) && (proc_type == util_type))
	      matched_procs.push(proc);
      }
      });
    }
  return matched_procs;
}


// This function gets a particular timelineElement. It will also handle
// creating the children for expand commands
function getElement(depth, text, full_text, type, num_levels, loader, 
                    tsv, _parent, children, enabled, visible) {

  // build the dummy element
  var element = {
    depth: depth,
    enabled: enabled,
    expanded: false,
    loaded: false,
    loader: loader,
    num_levels: num_levels,
    num_levels_ready: num_levels,
    children: [],
    parent: _parent,
    text: text,
    full_text: full_text,
    type: type,
    tsv: tsv,
    visible: visible,
    selected: true
  };
  // Create child elements. Child elements will be enabled, but they will be
  // invisible and unloaded. They will be loaded in when expanded
  if (children != undefined) {
    children.forEach(function(child) {
      var util_name = "node " + child;
      var child_element = getElement(depth + 1, util_name, undefined, 
                                     "util", num_levels, loader,
                                     "tsv/" + child + "_util.tsv",
                                     element, util_files[child], true, false);
      element.children.push(child_element);
    });
  }
  if ((type == "proc") && (_parent != undefined)) {
      var kind = get_kind(_parent.text);
      if (kind in state.processor_kinds) {
	  var r = /\d+/;
	  if (is_proc_string(full_text) && text.match(r)) {
	      var kind_proc = state.processor_kinds[kind];
	      var val = parseInt(text.match(r));
	      if ((kind_proc.low_range < 0) || (kind_proc.low_range > val))
		  kind_proc.low_range = val;
	      if (kind_proc.high_range < val)
		  kind_proc.high_range = val;
	  }
      }
  }

  // Util graphs will have processors as their children as well.
  if (type == "util") {
    // get the children for the util view
    var proc_children = getProcessors(text);
    var count = 1;
    var node_id = get_node_id(text);
    var kind = get_kind(text);
    // add to processor kinds: union of all processor kinds
    if (!(kind in state.processor_kinds)) {
      var pkind = {
      id: kind,
      value: kind,
      count: proc_children.length,
      low_range: -1,
      high_range: -1,
      text: kind
      };
      state.processor_kinds[kind] = pkind;
    }
    if (state.processor_kinds[kind].count < proc_children.length)
      state.processor_kinds[kind].count = proc_children.length;

    proc_children.forEach(function(proc_child) {
      var child_element = getElement(depth + 1, proc_child.text,
                                     proc_child.full_text, "proc", 
                                     proc_child.height, load_proc_timeline,
                                     proc_child.tsv, element, undefined, 
                                     true, false);
      element.children.push(child_element);
    });
  }

  return element;
}


function calculateLayout() {

  // First element in the layout will be the all_util. All first-level
  // elements will start off not enabled, and will be uncollapsed later
  // programmatically
  var num_nodes = Object.keys(util_files).length;
  if (num_nodes > 1) {
    var proc_kinds = util_files["all"];
    proc_kinds.forEach(function(name) {
      var kind = "(" + get_kind(name) + ")";
      var util_name = "all nodes " + kind;
      var kind_element = getElement(0, util_name, undefined, "util", 
                                     constants.util_levels, load_util,
                                     "tsv/" + name + "_util.tsv",
                                     undefined, util_files[kind], false, true);
      state.layoutData.push(kind_element);
    });
  }
  
  var seen_nodes = [];
  state.processors.forEach(function(proc) {
    // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
    var proc_regex = /(Processor|Memory) 0x1(e|d)([a-fA-f0-9]{4})[a-fA-f0-9]{10}$/;
    // Memory Channel
    var proc_channel_regex = /(Memory) 0x1(e)([a-fA-f0-9]{4})[a-fA-f0-9]{10}.*Channel/;
    var proc_channel_match  = proc_channel_regex.exec(proc.full_text);
    var proc_match = proc_regex.exec(proc.full_text);
    proc_match = proc_match || proc_channel_match;
    if (proc_match) {
      var node_id = parseInt(proc_match[3], 16);
      if (!(node_id in seen_nodes)) {
        seen_nodes[node_id] = 1;
        if (node_id in util_files) {
            var proc_kinds = util_files[node_id];
            proc_kinds.forEach(function(kind) {
		var util_name = "node " + kind;
		var kind_element = getElement(0, util_name, undefined, "util",
                                              constants.util_levels, load_util,
                                              "tsv/" + kind + "_util.tsv",
                                              undefined, util_files[kind],
                                              false, true);
		state.layoutData.push(kind_element);
            });
	}
      }
    }
      else {
	  state.layoutData.push(getElement(0, proc.text, proc.full_text, "proc",
					   proc.height, load_proc_timeline,
					   proc.tsv, undefined, undefined,
					   false, true));
      }
  });
  state.num_nodes = seen_nodes.length;
}

function getElemCoords(elems) {
  var proc = elems[0].proc;
  var level = elems[0].level;
  var startX = convertToPos(state, elems[0].start);
  var endX = convertToPos(state, elems[elems.length-1].end);
  var endBase = +proc.base;
  var endLevel = endBase + level;
  var y = dependencyLineLevelCalculator(endLevel);
  return {startX: startX, endX: endX, y: y};
}


function addLine(group, x1, x2, y1, y2, color, dashed) {
  var line;
  if (dashed) {
    line = group.append("line")
      .attr("x1", x1)
      .attr("y1", y1)
      .attr("x2", x2)
      .attr("y2", y2)
      .style("stroke", color)
      .style("stroke-dasharray", "3,3")
      .style("stroke-width", "1px");
  } else {
    line = group.append("line")
      .attr("x1", x1)
      .attr("y1", y1)
      .attr("x2", x2)
      .attr("y2", y2)
      .style("stroke", color)
      .style("stroke-width", "1px");
  }

  var slope = (y2 - y1) / (x2 - x1);
  var intercept = y1 - (slope * x1);
  var triangleRotation =  Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI - 90;

  group.append("line")
    .attr("x1", x1)
    .attr("y1", y1)
    .attr("x2", x2)
    .attr("y2", y2)
    .style("stroke", "transparent")
    .style("stroke-width", "20px")
    .on("mouseover", function() { 
      group.append("g")
      .attr("class", "marker")
      .append("path")
        .attr("d", d3.svg.symbol().type("triangle-down").size(20))
        .style("stroke", color)
        .style("fill", color);
    })
    .on("mousemove", function() {
      var marker = group.select("g.marker");
      var px = d3.mouse(this)[0] - 10;
      var py = d3.mouse(this)[1] - ((slope > 0 ? 1 : -1) * 20);
      if (px < Math.min(x1, x2)) {
        px += 20;
      }
      if (py < Math.min(y1, y2)) {
        py += 40;
      }

      // some linear algebra here
      var cx = (slope * py + px - slope * intercept) / (slope * slope + 1);
      var cy = (slope * slope * py + slope * px + intercept) / (slope * slope + 1);

      marker.attr("transform", "translate(" + cx + "," + cy + ") rotate(" + triangleRotation + ")");
    })
    .on("mouseout", function() {
      group.select("g.marker").remove();
    });

  group.append("circle")
    .attr("cx", x1)
    .attr("cy", y1)
    .attr("fill", "white")
    .attr("stroke", "black")
    .attr("r", 2.5)
    .style("stroke-width", "1px");

  group.append("circle")
    .attr("cx", x2)
    .attr("cy", y2)
    .attr("fill", "white")
    .attr("stroke", "black")
    .attr("r", 2.5)
    .style("stroke-width", "1px");
}


function drawCriticalPath() {
  state.timelineSvg.selectAll("g.critical_path_lines").remove();
  if (state.critical_path == undefined || !state.display_critical_path) {
    return;
  }
  var depGroup = state.timelineSvg.append("g")
      .attr("class", "critical_path_lines");
  state.critical_path.forEach(function(op) {
    var proc = base_map[op[0] + "," + op[1]]; // set in calculateBases
    if (proc != undefined && proc.visible && proc.enabled && proc.loaded && proc.selected) {
      var elems = prof_uid_map[op[2]];
      if (elems != undefined) {
        var coords = getElemCoords(elems)
        var startX = coords.startX;
        var endX = coords.endX;
        var y = coords.y;
        elems[0].out.forEach(function(dep) {
          if (dep[2] in state.critical_path_prof_uids) {
            var depElems = prof_uid_map[dep[2]];
            if (depElems != undefined) {
              var depProc = depElems[0].proc;
              // is the proc visible?
              if (depProc.visible && depProc.enabled && depProc.loaded) { 
                var depCoords = getElemCoords(depElems);
                var depX = depCoords.endX;
                var depY = depCoords.y;
                addLine(depGroup, depX, startX, depY, y, "grey", false);
              }
            }
          }
        });
        // Draw parent-child lines
        var lastChildCoords = undefined;
        var firstChildCoords = undefined;
        elems[0].children.forEach(function(child) {
          if (child[2] in state.critical_path_prof_uids) {
            var childElems = prof_uid_map[child[2]];
            var childCoords = getElemCoords(childElems);
            if (childElems !== undefined) {
              if (lastChildCoords === undefined || 
                  childCoords.startX > lastChildCoords.startX) {
                lastChildCoords = childCoords;
              }
              if (firstChildCoords === undefined || 
                  childCoords.endX < firstChildCoords.endX) {
                firstChildCoords = childCoords;
              }
            }
          }
        });
        if (firstChildCoords !== undefined) {
          addLine(depGroup, startX, firstChildCoords.startX, 
                  y, firstChildCoords.y, "grey", true);
        }
        if (lastChildCoords !== undefined) {
          addLine(depGroup, lastChildCoords.endX, endX,
                   lastChildCoords.y, y, "grey", true);
        }
      }
    }
  });
}

function drawDependencies() {
  state.timelineSvg.select("g.dependencies").remove();
  var timelineEvent = state.dependencyEvent;
  if (timelineEvent == undefined || 
      !timelineEvent.proc.visible || !timelineEvent.proc.enabled) {
    return; // don't draw in this case, the src isn't present
  }
  var srcElems = prof_uid_map[timelineEvent.prof_uid];
  timelineEvent = srcElems[0]; // only the first elem will have deps and children
  var srcCoords = getElemCoords(srcElems);
  var srcStartX = srcCoords.startX;
  var srcEndX = srcCoords.endX;
  var srcY = srcCoords.y;
  var depGroup = state.timelineSvg.append("g")
      .attr("class", "dependencies");

  var addDependency = function(dep, dir, dashed) {
    var depProc = base_map[dep[0] + "," + dep[1]]; // set in calculateBases
    if (depProc.visible && depProc.enabled && depProc.selected) {
      var depElems = prof_uid_map[dep[2]];
      if (depElems != undefined) {
        console.log(dir);
        var depCoords = getElemCoords(depElems);
        var dstY = depCoords.y;
        if (dir === "in") {
          addLine(depGroup, srcEndX, depCoords.startX, srcY, dstY, "black", dashed);
        } else if (dir === "out") {
          addLine(depGroup, depCoords.endX, srcStartX, dstY, srcY, "black", dashed);
        } else if (dir === "parent") {
          addLine(depGroup, depCoords.startX, srcStartX, dstY, srcY, "black", dashed);
          addLine(depGroup, srcEndX, depCoords.endX, srcY, dstY, "black", dashed);
        } else if (dir === "child") {
          addLine(depGroup, srcStartX, depCoords.startX, srcY, dstY, "black", dashed);
          addLine(depGroup, depCoords.endX, srcEndX, dstY, srcY, "black", dashed);
        }

      }
    }
  }

  if (state.drawDeps) {
    timelineEvent.in.forEach(function(dep) {addDependency(dep, "in", false)});
    timelineEvent.out.forEach(function(dep) {addDependency(dep, "out", false)});
  }
  if (state.drawChildren) {
    timelineEvent.parents.forEach(function(dep) {addDependency(dep, "parent", true)});
    timelineEvent.children.forEach(function(dep) {addDependency(dep, "child", true)});
  }
}

function timelineEventsExistAndEqual(a, b) {
  if (a == undefined || b == undefined) {
    return false;
  }
  return (a.proc.base == b.proc.base) && (a.prof_uid == b.prof_uid);
}

function timelineElementStrokeCalculator(elem) {
  if (timelineEventsExistAndEqual(elem, state.dependencyEvent)) {
    return true;
  } else if (state.display_critical_path &&
             elem.prof_uid in state.critical_path_prof_uids) {
    return true;
  }
  return false;
}

function timelineEventMouseDown(_timelineEvent) {
  // only the first event of this uid will have information
  var timelineEvent = prof_uid_map[_timelineEvent.prof_uid][0];
  var hasDependencies = ((timelineEvent.in.length != 0) || 
                         (timelineEvent.out.length != 0));

  if (hasDependencies) {
    if (timelineEventsExistAndEqual(timelineEvent, state.dependencyEvent)) {
      if (d3.event.button === 0) {
        state.drawDeps = !state.drawDeps;
      } else if (d3.event.button === 2) {
        state.drawChildren = !state.drawChildren;
      }
      if (state.drawDeps === false && state.drawChildren === false) {
        state.dependencyEvent = undefined;
      }
    } else {
      timelineEvent.in.concat(timelineEvent.out).forEach(function(dep) {
        expandByNodeProc(dep[0], dep[1])
      });
      state.dependencyEvent = timelineEvent;
      if (d3.event.button === 0) {
        state.drawDeps = true;
      } else if (d3.event.button === 2) {
        state.drawChildren = true;
      }
    }
    redraw();
  }
}

function drawTimeline() {
  showLoaderIcon()

  updateURL(state.zoom, state.scale);
  var timelineGroup = state.timelineSvg.select("g#timeline");
  timelineGroup.selectAll("rect").remove();
  timelineGroup.selectAll("text").remove();
  var timeline = timelineGroup.selectAll("rect")
    .data(state.dataToDraw, function(d) { return d.proc.base + "-" + d.id; });
  var timelineText = timelineGroup.selectAll("text")
    .data(state.memoryTexts);
  var mouseOver = getMouseOver();

  timeline.enter().append("rect");

  timeline
    .attr("id", function(d) { return "block-" + d.proc.base + "-" + d.id; })
    .attr("x", function(d) { return convertToPos(state, d.start); })
    .attr("y", timelineLevelCalculator)
    .style("fill", function(d) {
      if (!state.searchEnabled ||
          searchRegex[currentPos].exec(d.title) == null)
        return d.color;
      else return "#ff0000";
    })
    .attr("width", function(d) {
      return Math.max(constants.min_feature_width, convertToPos(state, d.end - d.start));
    })
    .attr("stroke", function(d) {
      if (timelineElementStrokeCalculator(d)) {
        return "red";
      } else {
        return "black";
      }
    })
    .attr("stroke-width", function(d) {
      if (timelineElementStrokeCalculator(d)) {
        return "1.5";
      } else {
        return "0.5";
      }
    })
    .attr("height", state.thickness)
    .style("opacity", function(d) {
      if (!state.searchEnabled || searchRegex[currentPos].exec(d.title) != null || searchRegex[currentPos].exec(d.initiation) != null) {
        return d.opacity;
      }
      else return 0.05;
    })
    .on("mouseout", function(d, i) { 
      if ((d.in.length != 0) || (d.out.length != 0)) {
        d3.select(this).style("cursor", "default");
      }
      state.timelineSvg.selectAll("#desc").remove();
    })
    .on("mousedown", timelineEventMouseDown)


  timelineText.enter().append("text");

  timelineText
    .attr("class", "timeline")
    .attr("y", timelineTextLevelCalculator)
    .attr("text-anchor", "start")
    .text(function(d) { 
       var sizeRegex = /Size=(.*)/;
       var match = sizeRegex.exec(d.title);
       return match[1];
    })
    .attr("visibility", function(d) {
      var textWidth = this.getComputedTextLength();
      var startX = convertToPos(state, d.start);
      var endX = convertToPos(state, d.end);
      var boxWidth = endX - startX;
      return boxWidth > textWidth ? "visible" : "hidden";
    })
    .attr("x", function(d) { 
      var textWidth = this.getComputedTextLength();
      var midPoint = convertToPos(state, d.start + (d.end - d.start) / 2); 
      midPoint -= textWidth / 2;
      return midPoint
    });

  drawUtil();
  timeline.on("mouseover", mouseOver);
  timeline.exit().remove();
  hideLoaderIcon();
}

function redraw() {
  if (state.numLoading == 0) {
    calculateBases();
    filterAndMergeBlocks(state);
    constants.max_level = calculateVisibileLevels();
    recalculateHeight();
    drawTimeline();
    drawDependencies();
    drawCriticalPath();
    drawLayout();
  }
}

function calculateVisibileLevels() {
  var levels = 0;
  state.flattenedLayoutData.forEach(function(elem) {
    if (elem.visible) {
      if (elem.enabled) {
	  if (state.ready_selected && elem.num_levels_ready != undefined)
	      levels += elem.num_levels_ready;
	  else
              levels += elem.num_levels;
      }
      levels += constants.elem_separation;
    }
  });
  return levels;
}

function is_proc(proc) {
  // the full text of this proc should start with 0x1d to be a processor
  return /0x1d/.exec(proc.full_text) !== undefined;
}

function is_proc_string(text)
{
    return /0x1d/.exec(text) != undefined;
}

function is_mem_string(text) {
    return /0x1e/.exec(text) != undefined;
}

function get_node_val(text) {
    var match = /\d+/.exec(text);
    if (match) {
	var val = parseInt(text.match(/\d+/));
	return val;
    }
    return match;
}

function get_node_id(text) {
  // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
  var proc_regex = /(Memory|Processor) 0x1(e|d)([a-fA-f0-9]{4})/;
  var proc_match = proc_regex.exec(text);
  // if there's only one node, then per-node graphs are redundant
  if (proc_match) {
    var node_id = parseInt(proc_match[3], 16);
    return node_id
  }
}

function get_proc_in_node(text) {
  // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
  var proc_regex = /Processor 0x1d[a-fA-f0-9]{11}([a-fA-f0-9]{3})/;
  var proc_match = proc_regex.exec(text);
  // if there's only one node, then per-node graphs are redundant
  if (proc_match) {
    var proc_in_node = parseInt(proc_match[1], 16);
    return proc_in_node;
  }
}

function calculateBases() {
  var base = 0;
  state.flattenedLayoutData.forEach(function(elem) {
    elem.base = base;
    var node_id = get_node_id(elem.full_text);
    var proc_in_node = get_proc_in_node(elem.full_text);
    if (node_id != undefined && proc_in_node != undefined) {
      base_map[(+node_id) + "," + (+proc_in_node)] = elem;
    }
    if (elem.visible) {
      if (elem.enabled) {
	  if (state.ready_selected && elem.num_levels_ready != undefined)
	      base += elem.num_levels_ready;
	  else
              base += elem.num_levels;
      }
      base += constants.elem_separation;
    }
  });
}

function expandHandler(elem, index) {
  elem.expanded = !elem.expanded;

  if (elem.expanded) {
    function expandChild(child) {
      child.visible = true;
      //child.enabled = true;
      if(!child.loaded) {
        showLoaderIcon();
        child.loader(child); // will redraw the timeline once loaded
      } else if (child.expanded) {
        child.children.forEach(expandChild);
      }
    }
    elem.children.forEach(expandChild);
  } else {
    function collapseChildren(child) {
      child.visible = false;
      child.children.forEach(collapseChildren);
    }
    elem.children.forEach(collapseChildren);
  }
  redraw();
}

// This handler is called when you collapse/uncollapse a row in the
// timeline
function collapseHandler(d, index) {
  d.enabled = !d.enabled;

  if (!d.loaded) {
    // should always be expanding here
    showLoaderIcon();
    //var elem = state.flattenedLayoutData[index];
    d.loader(d); // will redraw the timeline once loaded
  } else {
    redraw();
  }
}

// This expands a particular element and its parents if necessary
function expandElementAndParents(elem) {
  if (elem !== undefined) {
    var elemPath = []; // path up to parent
    var curElem = elem;
    while (curElem != undefined) {
      elemPath.push(curElem);
      curElem = curElem.parent;
    }

    for (var i = elemPath.length - 1; i >= 0; --i) {
      var elem = elemPath[i];
      if (!elem.expanded) {
        expandHandler(elem);
      }
      // uncollapse if necessary
      if ((i == 0) && !elem.enabled) {
        collapseHandler(elem);
      }
    }
  }
}


function expandByNodeProc(node, proc) {
  // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
  // ugh why isn't there string formatting
  var nodeHex = ("0000" + node.toString(16)).slice(-4);
  var procHex = ("000" + proc.toString(16)).slice(-3);
  var expectedTitle = "0x1d" + nodeHex + "0000000" + procHex;
  var matchedElement = undefined;
  for (var i = 0; i < state.flattenedLayoutData.length; i++) {
    var timelineElement = state.flattenedLayoutData[i];
    if ((timelineElement.full_text != undefined) &&
        (timelineElement.full_text.indexOf(expectedTitle) !=-1)) {  
      matchedElement = timelineElement;
      break;
    }
  }
  expandElementAndParents(matchedElement)
}


function lineLevelCalculator(timelineElement) {
  var level = timelineElement.base + 1;
  if (timelineElement.enabled) {
      if (state.ready_selected && timelineElement.num_levels_ready != undefined)
	  level += timelineElement.num_levels_ready;
      else
          level += timelineElement.num_levels;
  }
    return constants.margin_top + level * state.thickness;
};

function utilLevelCalculator(timelineElement) {
  var level = timelineElement.base;
    if (timelineElement.enabled) {
	if (state.ready_selected && timelineElement.num_levels_ready != undefined)
	    level += timeLineElement.num_levels_ready;
	else
            level +=  timeLineElement.num_levels;
    }
    return constants.margin_top + level * state.thickness;
};


function timelineLevelCalculator(timelineEvent) {
  return constants.margin_top +
         (timelineEvent.proc.base + timelineEvent.level) * state.thickness;
};

function timelineTextLevelCalculator(timelineEvent) {  
  return constants.margin_top + 
         (timelineEvent.proc.base + timelineEvent.level + 0.75) * state.thickness;
};

function dependencyLineLevelCalculator(level) {
  return constants.margin_top + ((level+0.5) * state.thickness);
}

function drawLayout() {
  d3.select("#processors").select("svg").remove();
  state.timelineSvg.select("g#lines").remove();

  var data = state.flattenedLayoutData;
  var svg = d3.select("#processors").append("svg")
    .attr("width", constants.margin_left)
    .attr("height", state.height);

  var namesGroup = svg.selectAll(".processors")
    .data(data)
    .enter().append("g");

  var names = namesGroup.append("text");

  var thickness = state.thickness;
  var xCalculator = function(d) { return (d.depth + 1) * 15; };
  names.attr("text-anchor", "start")
    .attr("class", "processor")
    .text(function(d) { return d.text; })
    .attr("x", xCalculator)
    .attr("y", lineLevelCalculator)
    .attr("visibility", function(elem) {
      return (elem.visible) ? "visible" : "hidden"  
    })
    .style("fill", "#000000")
    .style("opacity", function(proc) {
      return (proc.enabled) ? 1 : 0.5;
    });


  // names.each(function(d) {
  //   var elem = d3.select(this);
  //   var text = d.text;
  //   var tokens = d.text.split(" to ");
  //   if (tokens.length == 1)
  //     elem.append("tspan").text(d.text);
  //   else {
  //     var source = tokens[0];
  //     var target = tokens[1].replace(" Channel", "");
  //     elem.append("tspan").text(source)
  //       .attr("x", xCalculator)
  //       .attr("dy", -10);
  //     elem.append("tspan").text("==> " + target)
  //       .attr("x", xCalculator)
  //       .attr("dy", 10);
  //   }
  // });
  names.on({
    "mouseover": function(d) {
      d3.select(this).style("cursor", "pointer")
      if (d.full_text != undefined) {
        var x = xCalculator(d);
        var y = lineLevelCalculator(d) - state.thickness;

        var overlaySvg = d3.select("#overlay").append("svg");
        var descView = overlaySvg.append("g").attr("id", "desc");
        
        var text = descView.append("text")
          .attr("x", x)
          .attr("y", y)
          .attr("text-anchor", "start")
          .attr("class", "desc")
          .text(unescape(escape(d.full_text)));

        var bbox = text.node().getBBox();
        var padding = 2;
        var rect = descView.insert("rect", "text")
            .attr("x", bbox.x - padding)
            .attr("y", bbox.y - padding)
            .attr("width", bbox.width + (padding*2))
            .attr("height", bbox.height + (padding*2))
            .style("fill", "#222")
            .style("opacity", "0.7");

        overlaySvg.attr("width", bbox.x + bbox.width + (padding*2))
                  .attr("height", bbox.y + bbox.height + (padding*2));
      }
    },
    "mouseout": function(d) {
      d3.select(this).style("cursor", "default")
      if (d.full_text != undefined) {
        d3.select("#overlay").selectAll("svg").remove();
      }
    },
    "click": collapseHandler
  });

  var expandableNodes = namesGroup.filter(function(elem) {
    return elem.children.length > 0;
  });

  var expand_group = expandableNodes.append("g");

  var expand_clickable = expand_group.append("circle")
                          .attr("fill", "transparent")
                          .attr("stroke", "transparent")
                          .attr("r", 8);
  var expand_icons = expand_group.append("path");
  var arc = d3.svg.symbol().type('triangle-down')
            .size(12);


    //.attr("transform", function(elem) {
    //  var x = constants.margin_left - 10;
    //  var y = lineLevelCalculator(elem);
    //  return "translate(" + x + ","  + y + ")";
    //})
  expand_icons.attr("class", "processor")
    .attr("d", arc)
    .attr("visibility", function(elem) {
      return (elem.visible) ? "visible" : "hidden"  
    })
    .style("fill", "#000000")
    .style("stroke", "#000000")
    .style("opacity", function(proc) {
      return (proc.enabled) ? 1 : 0.5;
    });

  expand_group.each(function(elem) {
    var path = d3.select(this);
    var x = elem.depth * 15 + 5;
    var y = lineLevelCalculator(elem) - 3;

    if (elem.expanded) {
      path.attr("transform", "translate(" + x + ", " + y + ") rotate(0)");
    } else {
      path.attr("transform", "translate(" + x + ", " + y + ") rotate(30)");
    }
 
  });


  expand_group.on({
    "mouseover": function(d) {
      d3.select(this).style("cursor", "pointer")
    },
    "mouseout": function(d) {
      d3.select(this).style("cursor", "default")
    },
    "click": expandHandler
  });


  var lines = state.timelineSvg
    .append("g")
    .attr("id", "lines");

  var thickness = state.thickness;
  lines.selectAll(".lines")
    .data(data)
    .enter().append("line")
    .attr("x1", 0)
    .attr("y1", lineLevelCalculator)
    .attr("x2", state.zoom * state.width)
    .attr("y2", lineLevelCalculator)
    .attr("visibility", function(elem) {
      return (elem.visible) ? "visible" : "hidden"  
    })
    .style("stroke", "#000000")
    .style("stroke-width", "2px")
    .style("opacity", function(proc) {
      return (proc.enabled) ? 1 : 0.5;
    });
}

function drawHelpBox() {
  var width = $(window).width();
  var height = $(window).height();
  var popUpSvg = d3.select("#pop-up").select("svg");

  var helpBoxGroup = popUpSvg.append("g");
  var helpBoxWidth = Math.min(450, width - 100);
  var helpTextOffset = 20;
  var helpBoxHeight = Math.min(helpMessage.length * helpTextOffset + 100,
                               height - 100);

  var boxStartX = (width - helpBoxWidth) / 2;
  var boxStartY = (height - helpBoxHeight) / 2;

  helpBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: helpBoxWidth,
        height: helpBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var helpText = helpBoxGroup.append("text")
    .attr("class", "help-box")
    .style("width", helpBoxWidth);
  var helpTitle = "Keyboard Shortcuts";
  helpText.append("tspan")
    .attr({ x: boxStartX + helpBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text(helpTitle);
  var off = 15;
  for (var i = 0; i < helpMessage.length; ++i) {
    helpText.append("tspan")
      .style("font-size", "12pt")
      .attr("text-anchor", "start")
      .attr({ x: boxStartX + 30, dy: off + helpTextOffset})
      .text(helpMessage[i]);
    off = 0;
  }
}

function evalExpandRequest(re) {
  var re = new RegExp(re);
  state.flattenedLayoutData.forEach(function(elem) {
    if (re.exec(elem.text)) {
      expandElementAndParents(elem);
    }
  });
  removePopUp();
  makeTimelineOpaque();
  setKeyHandler(defaultKeydown);
  turnOnMouseHandlers();
}

function drawExpandBox() {
  var width = $(window).width();
  var height = $(window).height();
  var popUpSvg = d3.select("#pop-up").select("svg");
  var expandBoxGroup = popUpSvg.append("g");
  var expandBoxWidth = Math.min(450, width - 100);
  var expandBoxHeight = Math.min(220, height - 100);

  var thicknessRatio = state.thickness / state.baseThickness;
  var boxStartX = (width - expandBoxWidth) / 2;
  var boxStartY = (height - expandBoxHeight) / 2;

  expandBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: expandBoxWidth,
        height: expandBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var expandText = expandBoxGroup.append("text")
    .attr("class", "expand-box")
    .style("width", expandBoxWidth);
  expandText.append("tspan")
    .attr({ x: boxStartX + expandBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text("expand");
  var expandInputWidth = expandBoxWidth - 40;
  var expandInputHeight = 50;
  var expandInputStartY = 65;


  expandBoxGroup.append("foreignObject")
    .attr({ x: boxStartX + 20, y: boxStartY + expandInputStartY,
            width: expandInputWidth, height: expandBoxHeight - expandInputStartY})
    .attr("text-anchor", "middle")
    .html(
        "<input type='text' placeholder='expand by regex' class='expand-box' style='height: "
        + expandInputHeight + "px; width: "
        + expandInputWidth + "px; font-size: 20pt; font-family: Consolas, monospace;'></input>"
        + "<div style='margin: 0 auto; margin-top: 10px; text-align: center;'>" 
          + "<text class='expand-box'>" 
            + "Presets"
          + "</text>"
        + "</div>"
        + "<div style='margin: 0 auto; margin-top: 10px; text-align: center;'>" 
          + "<input class='button' type='button' value='CPUs' onclick='evalExpandRequest(\"CPU\")'></input>" 
          + "<input class='button' type='button' value='GPUs' onclick='evalExpandRequest(\"GPU\")'></input>" 
          + "<input class='button' type='button' value='Utilities' onclick='evalExpandRequest(\"Utility\")'></input>" 
          + "<input class='button' type='button' value='IOs' onclick='evalExpandRequest(\"IO\")'></input>" 
          + "<input class='button' type='button' value='Memories' onclick='evalExpandRequest(\"Memory\")'></input>" 
          + "<input class='button' type='button' value='Channels' onclick='evalExpandRequest(\"Channel\")'></input>"
        + "</div>"
        );
  $("input.expand-box").focus();
}

function drawSearchBox() {
  var width = $(window).width();
  var height = $(window).height();
  var popUpSvg = d3.select("#pop-up").select("svg");
  var searchBoxGroup = popUpSvg.append("g");
  var searchBoxWidth = Math.min(450, width - 100);
  var searchBoxHeight = Math.min(250, height - 100);

  var thicknessRatio = state.thickness / state.baseThickness;
  var boxStartX = (width - searchBoxWidth) / 2;
  var boxStartY = (height - searchBoxHeight) / 2;

  searchBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: searchBoxWidth,
        height: searchBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var searchText = searchBoxGroup.append("text")
    .attr("class", "search-box")
    .style("width", searchBoxWidth);
  searchText.append("tspan")
    .attr({ x: boxStartX + searchBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text("Search");
  var searchInputWidth = searchBoxWidth - 40;
  var searchInputHeight = 50;
  searchBoxGroup.append("foreignObject")
    .attr({ x: boxStartX + 20, y: boxStartY + 150,
            width: searchInputWidth, height: searchInputHeight})
    .attr("text-anchor", "middle")
    .html("<input type='text' class='search-box' style='height: "
        + searchInputHeight + "px; width: "
        + searchInputWidth + "px; font-size: 20pt; font-family: Consolas, monospace;'></input>");
  $("input.search-box").focus();
}

function drawSearchHistoryBox() {
  var width = $(window).width();
  var height = $(window).height();
  var popUpSvg = d3.select("#pop-up").select("svg");

  var historyBoxGroup = popUpSvg.append("g");
  var historyBoxWidth = Math.min(450, width - 100);
  var historyBoxHeight = Math.min(350, height - 100);

  var boxStartX = (width - historyBoxWidth) / 2;
  var boxStartY = (height - historyBoxHeight) / 2;

  historyBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: historyBoxWidth,
        height: historyBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var historyText = historyBoxGroup.append("text")
    .attr("class", "history-box")
    .style("width", historyBoxWidth);
  var historyTitle = "Search History";
  historyText.append("tspan")
    .attr({ x: boxStartX + historyBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text(historyTitle);
  if (searchRegex != null) {
    var off = 15;
    var id = 1;
    for (var i = 0; i < sizeHistory; ++i) {
      var pos = (nextPos + i) % sizeHistory;
      var regex = searchRegex[pos];
      if (regex != null) {
        if (pos == currentPos) prefix = ">>> ";
        else prefix = id + " : ";
        historyText.append("tspan")
          .attr("text-anchor", "start")
          .attr({ x: boxStartX + 30, dy: off + 25})
          .text(prefix + regex.source);
        off = 0;
        id++;
      }
    }
  }
}

function updateURL() {
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  var start_time = convertToTime(state, windowStart);
  var end_time = convertToTime(state, windowEnd);
  var url = window.location.href.split('?')[0];
  url += "?start=" + start_time;
  url += "&end=" + end_time;
  url += "&collapseAll=" + state.collapseAll;
  url += "&resolution=" + state.resolution;
  if (state.searchEnabled)
    url += "&search=" + searchRegex[currentPos].source;
  window.history.replaceState("", "", url);
}

function adjustZoom(newZoom, scroll) {
  var prevZoom = state.zoom;
  state.zoom = Math.round(newZoom * 10) / 10;

  var svg = d3.select("#timeline").select("svg");

  svg.attr("width", state.zoom * state.width)
     .attr("height", state.height);

  var timelineGroup = svg.select("g#timeline");
  //timelineGroup.selectAll("rect").remove();

  svg.select("g#lines").selectAll("line")
    .attr("x2", state.zoom * state.width);
  svg.selectAll("#desc").remove();
  svg.selectAll("g.locator").remove();
  d3.select("#overlay").selectAll("svg").remove();

  if (scroll) {
    var paneWidth = $("#timeline").width();
    var pos = ($("#timeline").scrollLeft() + paneWidth / 2) / prevZoom;
    // this will trigger a scroll event which in turn redraws the timeline
    $("#timeline").scrollLeft(pos * state.zoom - state.width / 2);
  } else {
    filterAndMergeBlocks(state);
    drawTimeline();
  }
  drawDependencies();
  drawCriticalPath();
}

function recalculateHeight() {
  // adjust the height based on the new thickness and max level
  state.height = constants.margin_top + constants.margin_bottom +
                 (state.thickness * constants.max_level);
  var util_height = constants.util_levels * state.thickness;
  state.y = d3.scale.linear().range([util_height, 0]);
  state.y.domain([0, 1]);

  d3.select("#processors").select("svg").remove();
  var svg = d3.select("#timeline").select("svg");
  svg.attr("width", state.zoom * state.width)
     .attr("height", state.height);
  var lines = state.timelineSvg.select("g#lines");
  lines.remove();

  svg.selectAll("#desc").remove();
  svg.selectAll("g.locator").remove();
  d3.select("#overlay").selectAll("svg").remove();

}

function adjustThickness(newThickness) {
  state.thickness = newThickness;
  recalculateHeight();
  drawTimeline();
  drawLayout();
}

function suppressdefault(e) {
  if (e.preventDefault) e.preventDefault();
  if (e.stopPropagation) e.stopPropagation();
}

function setKeyHandler(handler) {
  $(document).off("keydown");
  $(document).on("keydown", handler);
}

function makeModalKeyHandler(validKeys, callback) {
  return function(e) {
    if (!e) e = event;
    var code = e.keyCode || e.charCode;
    if (!(e.ctrlKey || e.metaKey || e.altKey)) {
      for (var i = 0; i < validKeys.length; ++i) {
        if (keys[code] == validKeys[i]) {
          callback(keys[code]);
          return false;
        }
      }
    }
    return true;
  }
}

function displayPopUp() {
    d3.select("#pop-up").append("svg")
                        .attr("width", $(window).width())
                        .attr("height", $(window).height());
}

function removePopUp() {
    var popUpSvg = d3.select("#pop-up").selectAll("svg").remove();
}


function defaultKeydown(e) {
  if (!e) e = event;

  var code = e.keyCode || e.charCode;
  var commandType = Command.none;
  var modifier = e.ctrlKey || e.metaKey || e.altKey;
  var multiFnKeys = e.metaKey && e.ctrlKey || e.altKey && e.ctrlKey;

  var translatedCode = keys[code];

  if (!modifier) {
    commandType = noModifierCommands[translatedCode];
  } else if (!multiFnKeys) {
    commandType = modifierCommands[translatedCode];
  } else {
    commandType = multipleModifierCommands[translatedCode];
  }

  if (commandType == undefined) {
    if (e.metaKey || e.altKey)
      state.rangeZoom = false;
    return true;
  }

  suppressdefault(e);
  if (commandType == Command.help) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    // make help box visible
    displayPopUp();
    drawHelpBox();
    setKeyHandler(makeModalKeyHandler(['/', 'esc'], function(key) {
      // make help box invisible
      removePopUp();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.search) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    displayPopUp();
    drawSearchBox();
    setKeyHandler(makeModalKeyHandler(['enter', 'esc'], function(key) {
      if (key == 'enter') {
        var re = $("input.search-box").val();
        if (re.trim() != "") {
          if (searchRegex == null) {
            searchRegex = new Array(sizeHistory);
            currentPos = -1;
            nextPos = 0;
          }
          currentPos = nextPos;
          nextPos = (nextPos + 1) % sizeHistory;
          searchRegex[currentPos] = new RegExp(re);
          state.searchEnabled = true;
        }
      }
      removePopUp();
      if (state.searchEnabled) {
        filterAndMergeBlocks(state);
        drawTimeline();
      }
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.expand) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    displayPopUp();
    drawExpandBox();
    setKeyHandler(makeModalKeyHandler(['enter', 'esc'], function(key) {
      if (key == 'enter') {
        var re = $("input.expand-box").val();
        if (re.trim() != "") {
          re = new RegExp(re);
          state.flattenedLayoutData.forEach(function(elem) {
            if (re.exec(elem.text)) {
              expandElementAndParents(elem);
            }
          });
        }
      }
      removePopUp();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.search_history) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    displayPopUp();
    drawSearchHistoryBox();
    setKeyHandler(makeModalKeyHandler(['h', 'esc'], function(key) {
      removePopUp();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }

  if (commandType == Command.zix) {
    var inc = 4.0;
    adjustZoom(state.zoom + inc, true);
  } else if (commandType == Command.zox) {
    var dec = 4.0;
    if (state.zoom - dec > 0)
      adjustZoom(state.zoom - dec, true);
  } else if (commandType == Command.zrx) {
    state.zoomHistory = Array();
    adjustZoom(1.0, false);
    $("#timeline").scrollLeft(0);
  } else if (commandType == Command.zux) {
    if (state.zoomHistory.length > 0) {
      var previousZoomHistory = state.zoomHistory.pop();
      adjustZoom(previousZoomHistory.zoom, false);
      $("#timeline").scrollLeft(previousZoomHistory.start);
    }
  } else if (commandType == Command.ziy)
    adjustThickness(state.thickness * 2);
  else if (commandType == Command.zoy)
    adjustThickness(state.thickness / 2);
  else if (commandType == Command.zry) {
    state.height = $(window).height() - constants.margin_bottom - constants.margin_top;
    state.thickness = state.height / constants.max_level;
    adjustThickness(state.thickness);
  }
  else if (commandType == Command.clear_search) {
    state.searchEnabled = false;
    searchRegex = null;
    filterAndMergeBlocks(state);
    drawTimeline();
  }
  else if (commandType == Command.toggle_search) {
    if (searchRegex != null) {
      state.searchEnabled = !state.searchEnabled;
      filterAndMergeBlocks(state);
      drawTimeline();
    }
  }
  else if (commandType == Command.previous_search ||
           commandType == Command.next_search) {
    if (state.searchEnabled) {
      var pos = commandType == Command.previous_search ?
                (currentPos - 1 + sizeHistory) % sizeHistory :
                (currentPos + 1) % sizeHistory;
      var sentinel = commandType == Command.previous_search ?
                (nextPos - 1 + sizeHistory) % sizeHistory : nextPos;
      if (pos != sentinel && searchRegex[pos] != null) {
        currentPos = pos;
        filterAndMergeBlocks(state);
        drawTimeline();
      }
    }
  }
  else if (commandType == Command.toggle_critical_path) {
    state.display_critical_path = !state.display_critical_path;
    if (state.display_critical_path) {
      state.critical_path.forEach(function(op) {
        expandByNodeProc(op[0], op[1]);
      });
    }
    redraw();
  }
  return false;
}

function defaultKeyUp(e) {
  if (!e) e = event;
  if (!(e.metaKey || e.altKey)) {
    state.rangeZoom = true;
  }
  return true;
}

function load_proc_timeline(proc) {
    var proc_name = proc.full_text;
    state.processorData[proc_name] = {};
    var num_levels_ready = proc.num_levels;
    if (state.ready_selected) {
	proc.num_levels_ready = proc.num_levels;
    }
  d3.tsv(proc.tsv,
    function(d, i) {
        var level = +d.level;
        var ready = +d.ready;
        var start = +d.start;
        var end = +d.end;
        var level_ready = +d.level_ready;
        var total = end - start;
        var _in = d.in === "" ? [] : JSON.parse(d.in)
        var out = d.out === "" ? [] : JSON.parse(d.out)
        var children = d.children === "" ? [] : JSON.parse(d.children)
        var parents = d.parents === "" ? [] : JSON.parse(d.parents)
        if (total > state.resolution) {
            return {
                id: i,
                level: level,
                level_ready: level_ready,
                ready: ready,
                start: start,
                end: end,
                color: d.color,
                opacity: d.opacity,
                initiation: d.initiation,
                title: d.title,
                in: _in,
                out: out,
                children: children,
                parents: parents,
                prof_uid: d.prof_uid,
                proc: proc
            };
        }
    },
    function(data) {
	var num_levels_ready=0
      // split profiling items by which level they're on
      for(var i = 0; i < data.length; i++) {
	  var d = data[i];
	  var level_sel=d.level;
          if (level_sel in state.processorData[proc_name]) {
              state.processorData[proc_name][level_sel].push(d);
          } else {
              state.processorData[proc_name][level_sel] = [d];
          }
	  if ((d.level_ready != undefined) && (d.level_ready != 0) &&
	      num_levels_ready < d.level_ready)
	      num_levels_ready = d.level_ready;

        if (d.prof_uid != undefined && d.prof_uid !== "") {
          if (d.prof_uid in prof_uid_map) {
            prof_uid_map[d.prof_uid].push(d);
          } else {
            prof_uid_map[d.prof_uid] = [d];
          }
        }
      }
    if (num_levels_ready > proc.num_levels)
	proc.num_levels_ready = num_levels_ready;
    else
	proc.num_levels_ready = proc.num_levels;
    proc.loaded = true;
    hideLoaderIcon();
    redraw();
    }
 );
}

function initTimelineElements() {
  var timelineGroup = state.timelineSvg.append("g")
      .attr("id", "timeline");

  $("#timeline").scrollLeft(0);
  parseURLParameters();

  var windowCenterY = $(window).height() / 2;
  $(window).scroll(function() {
      $("#loader-icon").css("top", $(window).scrollTop() + windowCenterY);
      $("#pop-up").css("top", $(window).scrollTop());
  });

  // set scroll callback
  var timer = null;
  $("#timeline").scroll(function() {
      if (timer !== null) {
        clearTimeout(timer);
      }
      timer = setTimeout(function() {
        filterAndMergeBlocks(state);
        drawTimeline();
      }, 100);
  });
  if (!state.collapseAll) {
    // initially load in all the cpu processors and the "all util"
    var util_regex = /node/;
    for (var i = 0; i < state.flattenedLayoutData.length; i++) {
      var timelineElement = state.flattenedLayoutData[i];
      if (timelineElement.type == "util" && 
          util_regex.exec(timelineElement.text) &&
          timelineElement.depth == 0) {
        collapseHandler(timelineElement, i);
      }
    }
  }
  turnOnMouseHandlers();
}


function load_ops_and_timeline() {
  d3.tsv("legion_prof_ops.tsv", // TODO: move to tsv folder
    function(d) {
      return d;
    },
    function(data) { 
      data.forEach(function(d) {
        state.operations[parseInt(d.op_id)] = d;
      });
      // initTimelineElements depends on the ops to be loaded
      initTimelineElements();
    }
  );
}

function load_procs(callback) {
  d3.tsv('legion_prof_processor.tsv',
    function(d) {
      return {
        full_text: d.full_text,
        text: d.text,
        height: +d.levels,
        tsv: d.tsv
      };
    },
    function(data) {
      var prevEnd = 0;
      var base = 0;
      data.forEach(function(proc) {
        proc.enabled = false;
        proc.loaded = false;
        proc.base = base;

        base += 2;
      });
      state.processors = data;
      calculateLayout();

      // flatten the layout and get the new calculated max num levels
      flattenLayout();
      calculateBases();
      drawLayout();
      createMenuList1();
      callback();
      
      // TODO: fix
      load_critical_path();
    }
  );
}

function getMinElement(arr, mapF) {
    if (arr.length === undefined || arr.length == 0) {
        throw "invalid array: " + arr;
    }
    var minElement = arr[0];
    if (mapF === undefined) {
        for (var i = 0; i < arr.length; i++) {
            if (arr[i] < minElement) {
                minElement = arr[i];
            }
        }
    } else {
        minValue = mapF(minElement);
        for (var i = 1; i < arr.length; i++) {
            var elem = arr[i];
            var value = mapF(elem);
            if (value < minValue) {
                minElement = elem;
                minValue = value;
            }
        }
    }
    return minElement;
}

function filterUtilData(timelineElem) {
  var data = state.utilData[timelineElem.tsv]; // TODO: attach to object
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  var start_time = convertToTime(state, windowStart - 100);
  var end_time = convertToTime(state, windowEnd + 100);

  var startIndex = timeBisector(data, start_time);
  var endIndex = timeBisector(data, end_time);

  startIndex = Math.max(0, startIndex - 1);
  endIndex = Math.min(data.length, endIndex + 2);
  var length = endIndex - startIndex;
  var newData = Array();
  // we will average points together if they are too close together
  var resolution = 1; // 3 pixel resolution
  var elem = {time: 0, count: 0};
  var startTime = data[startIndex].time;
  var startX = convertToPos(state, startTime);
  var endTime = 0;

  for (var i = startIndex; i < endIndex - 1; i++) {
    endTime = data[i+1].time;
    elem.count += data[i].count * (endTime - data[i].time);
    //elem.time += data[i].time;
    var endX = convertToPos(state, endTime);
    if ((endX - startX) >= resolution || i == (endIndex - 2)) {
      var totalTime = endTime - startTime;
      elem.count /= totalTime;
      //avgElem.time /= windowSize;
      elem.time = startTime;
      newData.push(elem);
      elem = {time: 0, count: 0};
      startX = endX;
      startTime = endTime;
    }
  }

  newData.push({time: endTime, count:0});

  return {
    text: timelineElem.text,
    base: timelineElem.base,
    num_levels: timelineElem.num_levels,
    num_levels_ready: timelineElem.num_levels_ready,
    data: newData
  };
}

function mouseover() { 
  var focus = state.timelineSvg.append("g")
    .attr("class", "focus");

  focus.append("circle")
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("r", 4.5);

  var utilDescView = state.timelineSvg.append("g")
                        .attr("class", "utilDesc");
  utilDescView.append("text")
    .attr("x", 7)
    .attr("y", -4.5)
    .attr("class", "desc")
    .attr("dy", ".35em");
  utilDescView.insert("rect", "text")
    .style("fill", "#222")
    .style("opacity", "0.7");
}

function mousemove(d, i) {
  var line = document.getElementById("util" + i);

  // offset for this particular line grapgh
  var base_y = +line.getAttribute("base_y");
  var px = d3.mouse(line)[0];
  var py = d3.mouse(line)[1] + base_y;
  
  var start = 0;
  var end = line.getTotalLength();
  var target = undefined;
  var pos = undefined;

  // https://bl.ocks.org/larsenmtl/e3b8b7c2ca4787f77d78f58d41c3da91
  while (true){
    target = Math.floor((start + end) / 2);
    pos = line.getPointAtLength(target);
    if ((target === end || target === start) && pos.x !== px) {
        break;
    }
    if (pos.x > px)      end = target;
    else if (pos.x < px) start = target;
    else break; //position found
  }


  var cx = pos.x;
  var cy = pos.y + base_y;

  var focus = state.timelineSvg.select("g.focus");
  focus.attr("transform", "translate(" + cx + "," + cy + ")");

  var utilDescView = state.timelineSvg.select("g.utilDesc");
  utilDescView.attr("transform", "translate(" + (px + 10) + "," + (py - 10) + ")");
  var text = utilDescView.select("text")

  text.text(Math.round(state.y.invert(pos.y) * 100) + "% utilization");

  var bbox = text.node().getBBox();
  var padding = 2;
  var rect = utilDescView.select("rect", "text");
  rect.attr("x", bbox.x - padding)
      .attr("y", bbox.y - padding)
      .attr("width", bbox.width + (padding*2))
      .attr("height", bbox.height + (padding*2));

  var timelineRight = $("#timeline").scrollLeft() + $("#timeline").width();
  var bboxRight = px + bbox.x + bbox.width;
  // If the box moves off the screen, nudge it back
  if (bboxRight > timelineRight) {
    var translation = -(bboxRight - timelineRight + 20);
    text.attr("transform", "translate(" + translation + ",0)");
    rect.attr("transform", "translate(" + translation + ",0)");
  }
}

// Get the data
function load_util(elem) {
  var util_file = elem.tsv;

  // exit early if we already loaded it
  if(state.utilData[util_file]) {
    elem.loaded = true;
    hideLoaderIcon();
    redraw();
    return;
  }

  d3.tsv(util_file,
    function(d) {
        return {
          time:  +d.time,
          count: +d.count
        };
    },
    function(error, data) {
      state.utilData[util_file] = data;
      elem.loaded = true;
      hideLoaderIcon();
      redraw();
    }
  );
}

function load_critical_path() {
  $.getJSON("json/critical_path.json", function(json) {
    state.critical_path = json.map(function(p) {return p.tuple});
    state.critical_path_prof_uids = {};
    state.critical_path.forEach(function(p) { 
      state.critical_path_prof_uids[p[2]] = 1;
    });
  });
}

function load_data() {
  load_procs(load_ops_and_timeline);
}

function initializeState() {
  var margin = constants.margin_left + constants.margin_right;
  state.width = $(window).width() - margin;
  state.height = $(window).height() - constants.margin_bottom;
  state.scale = state.width / (constants.end - constants.start);
  state.zoom = 1.0;
  state.zoomHistory = Array();
  state.numLoading = 0;
  state.dependencyEvent = undefined;
  state.drawDeps = false;
  state.drawChildren = false;

  state.layoutData = [];
  state.flattenedLayoutData = [];

  state.profilingData = {};
  state.processorData = {};
  state.utilData = [];

  state.operations = {};
  state.thickness = 20; //Math.max(state.height / constants.max_level, 20);
  state.baseThickness = state.height / constants.max_level;
  state.height = constants.max_level * state.thickness;
  state.resolution = 10; // time (in us) of the smallest feature we want to load
  state.searchEnabled = false;
  state.rangeZoom = true;
  state.collapseAll = false;
  state.display_critical_path = false;
  state.processor_kinds = {};

  // display task ready state?
  state.ready_selected = false;
  // TODO: change this
  state.x = d3.scale.linear().range([0, state.width]);
  state.y = d3.scale.linear().range([constants.util_levels * state.thickness, 0]);
  state.y.domain([0, 1]);

  setKeyHandler(defaultKeydown);
  $(document).on("keyup", defaultKeyUp);
  state.timelineSvg = d3.select("#timeline").select("svg").remove();
  state.timelineSvg = d3.select("#timeline").append("svg")
    .attr("width", state.zoom * state.width)
    .attr("height", state.height);

  state.loaderSvg = d3.select("#loader-icon").select("svg").remove();
  state.loaderSvg = d3.select("#loader-icon").append("svg")
    .attr("width", "40px")
    .attr("height", "40px");

  d3.select("#processors").select("svg").remove();

  drawLoaderIcon();
  drawHelpBox();
  load_data();
}

function load_util_json(callback) {
  $.getJSON("json/utils.json", function(json) {
    util_files = json;
    callback();
  });
}

// Loads constants such as max_level, start/end time, and number of levels
// of a util view
function load_scale_json(callback) {
  $.getJSON("json/scale.json", function(json) {
    // add the read-in constants
    $.each(json, function(key, val) {
      constants[key] = val;
    });
    load_util_json(callback);
  });
}

function load_op_deps_json(callback) {
  $.getJSON("json/op_dependencies.json", function(json) {
    op_dependencies = json;
    load_scale_json(callback);
  });
}

function load_jsons(callback) {
  load_scale_json(callback);
}

function main() {
  load_jsons(initializeState);
}

main();
