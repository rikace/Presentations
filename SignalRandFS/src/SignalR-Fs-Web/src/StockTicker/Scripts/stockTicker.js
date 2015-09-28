/// <reference path="../Scripts/jquery-1.10.2.js" />
/// <reference path="../Scripts/jquery.signalR-2.1.1.js" />
/// <reference path="smoothieChart.js" />

if (!String.prototype.supplant) {
    String.prototype.supplant = function (o) {
        return this.replace(/{([^{}]*)}/g,
            function (a, b) {
                var r = o[b];
                return typeof r === 'string' || typeof r === 'number' ? r : a;
            }
        );
    };
}

// A simple background color flash effect that uses jQuery Color plugin
jQuery.fn.flash = function (color, duration) {
    var current = this.css('backgroundColor');
    this.animate({ backgroundColor: 'rgb(' + color + ')' }, duration / 2)
        .animate({ backgroundColor: current }, duration / 2);
};

$(function () {
    var smoothie = new SmoothieChart();
    var colors = [];
    colors.push({ strokeStyle: 'rgb(0, 255, 0)', fillStyle: 'rgba(0, 255, 0, 0.4)', lineWidth: 3 });
    colors.push({ strokeStyle: 'rgb(255, 0, 255)', fillStyle: 'rgba(255, 0, 255, 0.3)', lineWidth: 3 });
    colors.push({ strokeStyle: 'rgb(125, 125, 0)', fillStyle: 'rgba(125, 125, 0, 0.2)', lineWidth: 3 });
    var lines = {};

    var ticker = $.connection.stockTicker, // the generated client-side hub proxy
        up = '▲',
        down = '▼',
        $stockTable = $('#stockTable'),
        $stockTableBody = $stockTable.find('tbody'),
        rowTemplate = '<tr data-symbol="{Symbol}"><td>{Symbol}</td><td>{Price}</td><td>{DayOpen}</td><td>{DayHigh}</td><td>{DayLow}</td><td><span class="dir {DirectionClass}">{Direction}</span> {Change}</td><td>{PercentChange}</td></tr>',
        $stockTicker = $('#stockTicker'),
        $stockTickerUl = $stockTicker.find('ul'),
        liTemplate = '<li data-symbol="{Symbol}"><span class="symbol">{Symbol}</span> <span class="price">{Price}</span> <span class="change"><span class="dir {DirectionClass}">{Direction}</span> {Change} ({PercentChange})</span></li>';

    function formatStock(stock) {
        return $.extend(stock, {
            Price: stock.Price.toFixed(2),
            PercentChange: (stock.PercentChange * 100).toFixed(2) + '%',
            Direction: stock.Change === 0 ? '' : stock.Change >= 0 ? up : down,
            DirectionClass: stock.Change === 0 ? 'even' : stock.Change >= 0 ? 'up' : 'down'
        });
    }

    function scrollTicker() {
        var w = $stockTickerUl.width();
        $stockTickerUl.css({ marginLeft: w });
        $stockTickerUl.animate({ marginLeft: -w }, 15000, 'linear', scrollTicker);
    }

    function stopTicker() {
        $stockTickerUl.stop();
    }

    function init() {
        ticker.server.getAllStocks();
        ticker.server.getMarketState();

        $('#addStock').click(function () {
            var connId = $.connection.hub.id;
           // $.post('http://localhost:48430/api/trading/Sell?connId=' + connId + '&ticker=ciao&quantity=4&price=9.5', function (data) {
                $.post('/api/trading/Sell?connId=' + connId + '&ticker=ciao&quantity=4&price=9.5', function (data) {
            });
        });

        // Wire up the buttons
        $("#open").click(function () {
            ticker.server.openMarket();
        });

        $("#close").click(function () {
            ticker.server.closeMarket();
        });
    }

    // Add client-side hub methods that the server will call
    $.extend(ticker.client, {
        setAllStocks: function (stocks) {
            $stockTableBody.empty();
            $stockTickerUl.empty();
            smoothie = new SmoothieChart({ grid: { strokeStyle: 'rgb(125, 0, 0)', maxValue: 125, minValue: 0, fillStyle: 'rgb(60, 0, 0)', lineWidth: 1, millisPerLine: 250, verticalSections: 6 } });

            var index = 0;
            $.each(stocks, function () {
                var stock = formatStock(this);
                lines[stock.Symbol] = new TimeSeries();
                smoothie.addTimeSeries(lines[stock.Symbol], colors[index]);
                $stockTableBody.append(rowTemplate.supplant(stock));
                $stockTickerUl.append(liTemplate.supplant(stock));
                index++;
            });
            smoothie.streamTo(document.getElementById("mycanvas"), 1000);


        },
        updateStockPrice: function (stock) {
            var displayStock = formatStock(stock),
                $row = $(rowTemplate.supplant(displayStock)),
                $li = $(liTemplate.supplant(displayStock)),
                bg = stock.LastChange < 0
                        ? '255,148,148' // red
                        : '154,240,117'; // green

            $stockTableBody.find('tr[data-symbol=' + stock.Symbol + ']')
                .replaceWith($row);
            $stockTickerUl.find('li[data-symbol=' + stock.Symbol + ']')
                .replaceWith($li);
            var valueStockChanged = (stock.Price * 100) / stock.DayOpen
            lines[stock.Symbol].append(new Date().getTime(), valueStockChanged.toFixed(2));
            $row.flash(bg, 1000);
            $li.flash(bg, 1000);
        },
        setMarketState: function (state) {
            if (state === 'Open') {
                $("#open").prop("disabled", true);
                $("#close").prop("disabled", false);
                scrollTicker();
            } else {
                $("#open").prop("disabled", false);
                $("#close").prop("disabled", true);
                stopTicker();
            }
        }
    });

    // Start the connection
    $.connection.hub.start().then(init);
});