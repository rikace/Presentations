using System;
using Common.Framework;
using Domain.Common;

public class TransactionHandler<TCommand, TCommandHandler>
    where TCommandHandler : ICommandHandler<TCommand>
    where TCommand : ICommand
{
    private readonly IUnitOfWork _unitOfWork;

    public TransactionHandler(IUnitOfWork unitOfWork)
    {
        _unitOfWork = unitOfWork;
    }

    public void Execute(TCommand command, TCommandHandler commandHandler)
    {
        try
        {
            commandHandler.Execute(command);
            _unitOfWork.Commit();
        }
        catch (Exception)
        {
            _unitOfWork.Rollback();
            throw;
        }
    }
}