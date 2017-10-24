using System;
using System.Data.Entity;
using System.Linq;
using System.Linq.Expressions;
using Domain.Entities;

namespace Domain.Common
{
    public class ContextRepository<T> : IRepository<T> where T : class, IEntity
    {
        private readonly DbSet<T> _set;

        public ContextRepository(DbContext ctx)
        {
            _set = ctx.Set<T>();
        }

        public IQueryable<T> FindAll()
        {
            return _set;
        }

        public IQueryable<T> Find(Expression<Func<T, bool>> predicate)
        {
            return _set.Where(predicate);
        }

        public T FindById(int id)
        {
            return _set.FirstOrDefault(f => f.Id == id);
        }

        public void Add(T newEntity)
        {
            _set.Add(newEntity);
        }

        public void Remove(T entity)
        {
            _set.Remove(entity);
        }
    }
}