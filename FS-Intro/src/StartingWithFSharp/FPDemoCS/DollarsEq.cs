using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FPDemoCS.Eq
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
        public override bool Equals(object obj)
        {
            var that = obj as Dollars;
            return
               that != null
               ? this.Amount == that.Amount
               : false;
        }
    }

}
