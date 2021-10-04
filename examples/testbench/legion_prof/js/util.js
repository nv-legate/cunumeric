//-----------------------------------------------------------------------------
//                                    UTILS
//-----------------------------------------------------------------------------

var helpMessage = [
  "Zoom-in (x-axis)   : Ctrl + / 4",
  "Zoom-out (x-axis)  : Ctrl - / 3",
  "Reset zoom (x-axis): Ctrl 0 / 0",
  "Undo zoom (x-axis) : u / U",
  "Zoom-in (y-axis)   : Ctrl-Alt + / 2",
  "Zoom-out (y-axis)  : Ctrl-Alt - / 1",
  "Reset zoom (y-axis): Ctrl-Alt 0 / `",
  "Range Zoom-in      : drag-select",
  "Measure duration   : Alt + drag-select",
  "Expand             : e / E",
  "Draw Critical Path : a / A",
  "Search             : s / S",
  "Search History     : h / H",
  "Previous Search    : p / P",
  "Next Search        : n / N",
  "Clear Search       : c / C",
  "Toggle Search      : t / T",
];

var Command = {
  none : 0,
  help : 1,
  zox : 2,
  zix : 3,
  zrx : 4,
  zux : 5,
  zoy : 6,
  ziy : 7,
  zry : 8,
  search : 9,
  clear_search : 10,
  toggle_search : 11,
  expand: 12,
  toggle_critical_path: 13,
  search_history : 14,
  previous_search : 15,
  next_search : 16
};

// commands without a modifier key pressed
var noModifierCommands = {
  '0': Command.zrx,
  '1': Command.zoy,
  '2': Command.ziy,
  '3': Command.zox,
  '4': Command.zix,
  'c': Command.clear_search,
  'e': Command.expand,
  'h': Command.search_history,
  'n': Command.next_search,
  'p': Command.previous_search,
  's': Command.search,
  'a': Command.toggle_critical_path,
  't': Command.toggle_search,
  'u': Command.zux,
  '/': Command.help
};

var modifierCommands = {
  '+': Command.zix,
  '-': Command.zox,
  '0': Command.zrx
}

var multipleModifierCommands = {
  '+': Command.ziy,
  '-': Command.zoy,
  '0': Command.zry
}

var keys = {
  13  : 'enter',
  27  : 'esc',
  48  : '0',
  49  : '1',
  50  : '2',
  51  : '3',
  52  : '4',
  61  : '+', // Firefox
  65  : 'a',
  67  : 'c',
  69  : 'e',
  72  : 'h',
  78  : 'n',
  80  : 'p',
  83  : 's',
  84  : 't',
  85  : 'u',
  173 : '-', // Firefox
  187 : '+',
  189 : '-',
  191 : '/',
  192 : '`'
};

/**
 * Function: convertToTime
 *
 * Description:
 *    Takes a value and converts it to time. The input 'x' can either be
 *    a single position on the timeline, or it can be a width.
 */
function convertToTime(state, x) {
  return x / state.zoom / state.scale;
}

/**
 * Function: convertToPos
 *
 * Description:
 *    Takes a time and converts it to a position in the window.
 *    The 'time' parameter MUST BE IN us
 */
function convertToPos(state, time) {
  return time * state.zoom * state.scale;
}

/**
 * Function: getTimeString
 *
 * Description:
 *    This function takes a time (in us) as well as a 'width'
 *    field. The 'width' field determines over what interval
 *    the time is being considered.
 *
 *    If the width is too large, the time will be converted
 *    to either ms or sec.
 *
 *    The function will return a string representation of the
 *    (possibly) scaled time at the end of the method
 */
function getTimeString(time, timeWidth) {
  var unit = "us";
  var scaledTime = Math.floor(time);
  if (timeWidth >= 100000) {
    var scaledTime = Math.floor(time / 1000);
    unit = "ms";
  } else if (timeWidth >= 100000000) {
    var scaledTime = Math.floor(time / 1000000);
    unit = "s";
  }
  return scaledTime + " " + unit;
}

// timeField should be 'end' for the left hand side and
// 'start' for the right hand side
function binarySearch(data, level, time, useStart) {
  var low = 0;
  var high = data[level].length;
  while (true) {
    // ugh is there a way to do integer division in javascrtipt?
    var mid = Math.floor((high - low) / 2) + low; 

    // In this case, we haven't found it. This is as close
    // as we were able to get.
    if (low == high || low == mid) {
      return low;
    } else if (high == mid) {
      return high;
    }
    var midTime;
    if (useStart) {
      midTime = data[level][mid].start;
    } else {
      midTime = data[level][mid].end;
    }

    if (midTime > time) {
      // need to look below
      high = mid;
    } else if (midTime < time) {
      // need to look above
      low = mid;
    } else {
      // Exact Match
      return mid;
    }
  }
}

dataReady = {};
function redoData(proc) {
    dataReady = {};
    var items =  state.processorData[proc.full_text];
    for (var level in items) {
	for (var j = 0; j< items[level].length; j++) {
	    d = items[level][j];
	    // ready state items
	    if (d.level_ready != undefined && d.level_ready != "" &&
		d.ready != undefined && d.ready != "")  {
		// if the level has already been switched, then don't do anything
		var level_to_set = d.level_ready;
		if (d.level_ready_set != undefined && d.level_ready_set == true)
		    level_to_set = d.level;
		var d_ready = {
                    id: d.id+items[level].length+1, // unique id
                    level: level_to_set,
		    level_ready: d.level,
                    ready: d.ready,
                    start: d.ready,
                    end: d.start,
                    color: d.color,
                    opacity: "0.45",
                    initiation: d.initiation,
                    title: d.title + " (ready)",
		    in: d.in,
                    out: d.out,
                    children: d.children,
                    parents: d.parents,
                    prof_uid: d.prof_uid,
                    proc: d.proc
		};
		// switch levels if we need to
		if (d.level_ready_set == undefined || d.level_ready_set == false)
		{
		    var level_tmp = d.level;
		    d.level = d.level_ready;
		    d.level_ready = level_tmp;
		    d.level_ready_set = true;
		}
		if (d.level_ready in dataReady){
		    dataReady[d.level_ready].push(d_ready);
		    dataReady[d.level_ready].push(d);
		}
		else {
		    dataReady[d.level_ready] = [d_ready];
		    dataReady[d.level_ready].push(d);
		}
	    }
	    // add items to the level
	    else {
		if (d.level in dataReady)
		    dataReady[d.level].push(d);
		else
		    dataReady[d.level] = [d];
	    }
	}
    }
}

function readyData(proc) {
    redoData(proc);
}

function filterAndMergeBlocks(state) {
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  state.dataToDraw = Array();
  state.memoryTexts = Array();
  var startTime = convertToTime(state, windowStart);
  var endTime = convertToTime(state, windowEnd);
  var min_feature_time = convertToTime(state, constants.min_feature_width);
  var min_gap_time = convertToTime(state, constants.min_gap_width);
  for (var index in state.flattenedLayoutData) {
    var timelineElement = state.flattenedLayoutData[index];
    if (timelineElement.type == "proc" && timelineElement.enabled && timelineElement.visible) {
      var items = state.processorData[timelineElement.full_text];
      var memoryRegex = /Memory/;
      var isMemory = memoryRegex.exec(timelineElement.text);
	if (state.ready_selected) {
	    readyData(timelineElement);
	    items = dataReady;
	}

      for (var level in items) {
        // gap merging below assumes intervals are sorted - do that first
        //items[level].sort(function(a,b) { return a.start - b.start; });

        // We will use binary search to limit the following section
        var startIndex = binarySearch(items, level, startTime, false);
        var endIndex = binarySearch(items, level, endTime, true);
        for (var i = startIndex; i <= endIndex; ++i) {
          var d = items[level][i];
          var start = d.start;
          var end = d.end;
	    // fix the level here
	    if (state.ready_selected == false && d.level_ready_set != undefined
		&& d.level_ready_set == true)
	    {
		// switch levels
		var level_tmp = d.level;
		d.level = d.level_ready;
		d.level_ready = level_tmp;
	        d.level_ready_set = false;
	    }
          // is this block too narrow?
          if ((end - start) < min_feature_time) {
            // see how many more after this are also too narrow and too close to us
            var count = 1;
            // don't do this if we're the subject of the current search and don't merge with
            // something that is
            if (!state.searchEnabled || searchRegex[currentPos].exec(d.title) == null) {
              while (((i + count) < items[level].length) && 
                     ((items[level][i + count].start - end) < min_gap_time) &&
                     ((items[level][i + count].end - items[level][i + count].start) < min_feature_time) &&
                     (!state.searchEnabled || searchRegex[currentPos].exec(items[level][i + count].title) == null)) {
                end = items[level][i + count].end;
                count++;
              }
            }
            // are we still too narrow?  if so, bloat, but make sure we don't overlap something later
            if ((end - start) < min_feature_time) {
              end = start + min_feature_time;                   
              if (((i + count) < items[level].length) && (items[level][i + count].start < end))
                end = items[level][i + count].start;
            }
            if (count > 1) {
              state.dataToDraw.push({
                id: d.id,
                prof_uid: d.prof_uid,
                proc: timelineElement,
                level: d.level,
                ready: d.ready,
                start: d.start,
                end: Math.round(end),
                color: "#808080",
                title: count + " merged tasks",
                in: [],
                out: [],
                children: [],
                parents: []
              });
              i += (count - 1);
            } else {
              var elem = {
                id: d.id,
                prof_uid: d.prof_uid,
                proc: timelineElement,
                level: d.level,
                ready: d.ready,
                start: d.start,
                end: d.end,
                color: d.color,
                initiation: d.initiation,
                title: d.title + " (expanded for visibility)",
                in: d.in,
                out: d.out,
                children: d.children,
                parents: d.parents
              }
              state.dataToDraw.push(elem);
              if (isMemory) {
                state.memoryTexts.push(elem);
              }
            }
          } else {
            var elem = {
              id: d.id,
              prof_uid: d.prof_uid,
              proc: timelineElement,
              level: d.level,
              ready: d.ready,
              start: d.start,
              end: d.end,
              opacity: d.opacity,
              color: d.color,
              initiation: d.initiation,
              title: d.title,
              in: d.in,
              out: d.out,
              children: d.children,
              parents: d.parents
            }
            state.dataToDraw.push(elem);
            if (isMemory) {
              state.memoryTexts.push(elem);
            }
          }
        }
      }
    }
  }
}
