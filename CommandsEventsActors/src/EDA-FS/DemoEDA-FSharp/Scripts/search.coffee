
class SearchViewModel
    constructor: () ->

        # Variables
        @auctions = ko.observableArray([])
        @categories = ko.observableArray([])
        @errorMessage = ko.observable('')
        @selectedCategory = ko.observable()
        @searchQuery = ko.observable('')

        @currentPage = ko.observable(0)
        @pageSize = ko.observable(5)


        # Computed properties
        @hasError = ko.computed =>
            @errorMessage().length > 0

        @hasSearchQuery = ko.computed =>
            @searchQuery()

        @hasResults = ko.computed =>
            @auctions().length > 0

        @resultsCount = ko.computed =>
            @auctions().length


        # Functions
        @load = (data) =>
            @auctions(data['Auctions'] || [])
            @searchQuery(data['SearchQuery'] || '')

        @search = =>
            $.ajax '/search',
                type: 'POST'
                data:
                    query: @searchQuery()
                    page: @currentPage
                    size: @pageSize
                error: => @errorMessage 'Error retrieving results'
                success: (data) => @load data


window.SearchViewModel = SearchViewModel

# Initialize autocomplete    
$ =>
    $('#search .search-query').data('source',
        (query, callback) =>
            category = $('#search [name=category]').val();
            $.getJSON '/autocomplete', 
                { query: query, category: category },
                callback
    )