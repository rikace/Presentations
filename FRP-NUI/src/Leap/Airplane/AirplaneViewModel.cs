using System;
using System.Linq;
using System.ComponentModel;
using Leap;
using static AirplaneListener;

namespace Airplane
{
    class AirplaneViewModel : INotifyPropertyChanged, IAirplaneViewModelInterface
    {
        AirplaneRotation airplane;
        public AirplaneViewModel()
        {
            airplane = new AirplaneRotation(this);
        }

        private double _xAngle;

        public double XAngle
        {
            get { return _xAngle; }
            set
            {
                _xAngle = value;
                OnPropertyChanged("XAngle");
            }
        }

        private double _yAngle;

        public double YAngle
        {
            get { return _yAngle; }
            set
            {
                _yAngle = value;
                OnPropertyChanged("YAngle");
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void OnPropertyChanged(string propertyName)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
            }
        }
    }
}
