(function() {
    var SearchViewModel,
        _this = this;

    SearchViewModel = (function() {

        function SearchViewModel() {
            var _this = this;
            this.auctions = ko.observableArray([]);
            this.categories = ko.observableArray([]);
            this.errorMessage = ko.observable('');
            this.selectedCategory = ko.observable();
            this.searchQuery = ko.observable('');
            this.currentPage = ko.observable(0);
            this.pageSize = ko.observable(5);
            this.hasError = ko.computed(function() {
                return _this.errorMessage().length > 0;
            });
            this.hasSearchQuery = ko.computed(function() {
                return _this.searchQuery();
            });
            this.hasResults = ko.computed(function() {
                return _this.auctions().length > 0;
            });
            this.resultsCount = ko.computed(function() {
                return _this.auctions().length;
            });
            this.load = function(data) {
                _this.auctions(data['Auctions'] || []);
                return _this.searchQuery(data['SearchQuery'] || '');
            };
            this.search = function() {
                return $.ajax('/search', {
                    type: 'POST',
                    data: {
                        query: _this.searchQuery(),
                        page: _this.currentPage,
                        size: _this.pageSize
                    },
                    error: function() {
                        return _this.errorMessage('Error retrieving results');
                    },
                    success: function(data) {
                        return _this.load(data);
                    }
                });
            };
        }

        return SearchViewModel;

    })();

    window.SearchViewModel = SearchViewModel;

    $(function() {
        return $('#search .search-query').data('source', function(query, callback) {
            var category;
            category = $('#search [name=category]').val();
            return $.getJSON('/autocomplete', {
                query: query,
                category: category
            }, callback);
        });
    });

}).call(this);