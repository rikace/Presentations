Public Interface IEventPublisherTest
End Interface

Interface IAsset
    Event ComittedChange(ByVal Success As Boolean)
    Property Division() As String
    Function GetID() As Integer
    Sub GetCiao()

End Interface