using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FPDemoCS
{
    public class Dollars
    {
        private readonly decimal _amount;
        public Dollars(decimal value)
        {
            _amount = value;
        }
        public decimal Amount
        {
            get { return _amount; }
        }
        public Dollars Times(decimal multiplier)
        {
            return new Dollars(this._amount * multiplier);
        }
    }

}
