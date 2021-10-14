// Setup the chart
const contribution = {"data": [{"key": "Nick Young", "y": 93556}, {"key": "Andrew Leathwick", "y": 18582}, {"key": "aleathwick", "y": 8037}]}
nv.addGraph(function() {
	var chart = nv.models.pieChart()
		.x(function(d) { return d.key })
		.y(function(d) { return d.y })
		.options({
                    "padAngle": 0.01,
                    "cornerRadius": 5
                });
	chart.pie.donutLabelsOutside(true).donut(true);

	d3.select('#chart_contribution svg').datum(contribution.data).call(chart);
	return chart;
});

const lines_stats = {"data": [{"key": "Nick Young", "values": [{"x": 1625356800000, "y": 0}, {"x": 1625961600000, "y": 0}, {"x": 1626566400000, "y": 3023}, {"x": 1627171200000, "y": 6774}, {"x": 1627776000000, "y": 6774}, {"x": 1628380800000, "y": 9049}, {"x": 1628985600000, "y": 51272}, {"x": 1629590400000, "y": 52003}, {"x": 1630195200000, "y": 56040}, {"x": 1630800000000, "y": 95248}, {"x": 1631404800000, "y": 102085}, {"x": 1632009600000, "y": 107443}, {"x": 1632614400000, "y": 115362}, {"x": 1633219200000, "y": 115362}, {"x": 1633824000000, "y": 115362}]}, {"key": "Andrew Leathwick", "values": [{"x": 1625356800000, "y": 6756}, {"x": 1625961600000, "y": 6756}, {"x": 1626566400000, "y": 6756}, {"x": 1627171200000, "y": 6756}, {"x": 1627776000000, "y": 6756}, {"x": 1628380800000, "y": 6904}, {"x": 1628985600000, "y": 11639}, {"x": 1629590400000, "y": 17629}, {"x": 1630195200000, "y": 41665}, {"x": 1630800000000, "y": 45938}, {"x": 1631404800000, "y": 47907}, {"x": 1632009600000, "y": 52916}, {"x": 1632614400000, "y": 79600}, {"x": 1633219200000, "y": 81224}, {"x": 1633824000000, "y": 81224}]}, {"key": "aleathwick", "values": [{"x": 1625356800000, "y": 4}, {"x": 1625961600000, "y": 4}, {"x": 1626566400000, "y": 4}, {"x": 1627171200000, "y": 4}, {"x": 1627776000000, "y": 4}, {"x": 1628380800000, "y": 4}, {"x": 1628985600000, "y": 4}, {"x": 1629590400000, "y": 4}, {"x": 1630195200000, "y": 4}, {"x": 1630800000000, "y": 4}, {"x": 1631404800000, "y": 4}, {"x": 1632009600000, "y": 4}, {"x": 1632614400000, "y": 4}, {"x": 1633219200000, "y": 4}, {"x": 1633824000000, "y": 15351}]}, {"key": "Others", "values": [{"x": 1625356800000, "y": 0}, {"x": 1625961600000, "y": 0}, {"x": 1626566400000, "y": 0}, {"x": 1627171200000, "y": 0}, {"x": 1627776000000, "y": 0}, {"x": 1628380800000, "y": 0}, {"x": 1628985600000, "y": 0}, {"x": 1629590400000, "y": 0}, {"x": 1630195200000, "y": 0}, {"x": 1630800000000, "y": 0}, {"x": 1631404800000, "y": 0}, {"x": 1632009600000, "y": 0}, {"x": 1632614400000, "y": 0}, {"x": 1633219200000, "y": 0}, {"x": 1633824000000, "y": 0}]}]}
// Setup the lines by author chart
nv.addGraph(function() {
	var chart = nv.models.lineChart()
		.useInteractiveGuideline(true);
	chart.yAxis.options({ "axisLabel": "Lines" });
	chart.xAxis
		.tickFormat(function(d) { return d3.time.format('%Y-%m')(new Date(d)); })
		.options({ "rotateLabels": -45 })

	d3.select('#chart_loc svg').datum(lines_stats.data).call(chart);
	return chart;
});

const commit_stats = {"data": [{"key": "Nick Young", "values": [[1625356800000, 0, 0], [1625961600000, 0, 0], [1626566400000, 2, 2], [1627171200000, 3, 1], [1627776000000, 3, 0], [1628380800000, 8, 5], [1628985600000, 13, 5], [1629590400000, 15, 2], [1630195200000, 17, 2], [1630800000000, 28, 11], [1631404800000, 32, 4], [1632009600000, 34, 2], [1632614400000, 38, 4], [1633219200000, 38, 0], [1633824000000, 38, 0]]}, {"key": "Andrew Leathwick", "values": [[1625356800000, 3, 0], [1625961600000, 3, 0], [1626566400000, 3, 0], [1627171200000, 3, 0], [1627776000000, 3, 0], [1628380800000, 5, 2], [1628985600000, 6, 1], [1629590400000, 9, 3], [1630195200000, 12, 3], [1630800000000, 19, 7], [1631404800000, 21, 2], [1632009600000, 25, 4], [1632614400000, 29, 4], [1633219200000, 33, 4], [1633824000000, 33, 0]]}, {"key": "aleathwick", "values": [[1625356800000, 2, 0], [1625961600000, 2, 0], [1626566400000, 2, 0], [1627171200000, 2, 0], [1627776000000, 2, 0], [1628380800000, 2, 0], [1628985600000, 2, 0], [1629590400000, 2, 0], [1630195200000, 2, 0], [1630800000000, 2, 0], [1631404800000, 2, 0], [1632009600000, 2, 0], [1632614400000, 2, 0], [1633219200000, 2, 0], [1633824000000, 18, 16]]}, {"key": "Others", "values": [[1625356800000, 0, 0], [1625961600000, 0, 0], [1626566400000, 0, 0], [1627171200000, 0, 0], [1627776000000, 0, 0], [1628380800000, 0, 0], [1628985600000, 0, 0], [1629590400000, 0, 0], [1630195200000, 0, 0], [1630800000000, 0, 0], [1631404800000, 0, 0], [1632009600000, 0, 0], [1632614400000, 0, 0], [1633219200000, 0, 0], [1633824000000, 0, 0]]}]}
// Setup the commits-by-author chart
nv.addGraph(function() {
	var chart = nv.models.lineChart()
		.x(function(d) { return d[0] })
		.y(function(d) { return d[1] })
		.useInteractiveGuideline(true);
	chart.yAxis.options({ "axisLabel": "Commits" });
	chart.xAxis
		.tickFormat(function(d) { return d3.time.format('%Y-%m')(new Date(d)); })
		.options({ "rotateLabels": -45 })

	d3.select('#chart_commits svg').datum(commit_stats.data).call(chart);
	return chart;
});

// Setup the streamgraph
nv.addGraph(function() {
	var chart = nv.models.stackedAreaChart()
		.x(function(d) { return d[0] })
		.y(function(d) { return d[2] })
		.options({
		        "useInteractiveGuideline": true,
		        "style": "stream-center",
                "showControls": false,
                "showLegend": false,
                });
	chart.yAxis.options({ "axisLabel": "Commits" });
	chart.xAxis
		.tickFormat(function(d) { return d3.time.format('%Y-%m')(new Date(d)); })
		.options({ "rotateLabels": -45 })

	d3.select('#chart_steam svg').datum(commit_stats.data).call(chart);
	return chart;
});

const domains = {"data": [{"key": "gmail.com", "y": 46}, {"key": "auckland.ac.nz", "y": 38}, {"key": "users.noreply.github.com", "y": 5}]}
// Setup the chart
nv.addGraph(function() {
	var chart = nv.models.pieChart()
		.x(function(d) { return d.key })
		.y(function(d) { return d.y })
		.options({
                "padAngle": 0.01,
                "cornerRadius": 5
            });
	chart.pie.donutLabelsOutside(true).donut(true);

	d3.select('#chart_domains svg').datum(domains.data).call(chart);
	return chart;
});