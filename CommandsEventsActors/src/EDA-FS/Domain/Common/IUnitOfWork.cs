using System;

namespace Domain.Common
{
    public interface IUnitOfWork : IDisposable
    {
        void Commit();
        void Rollback();
    }
}