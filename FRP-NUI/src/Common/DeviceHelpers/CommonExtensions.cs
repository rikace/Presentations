using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Kinect;

namespace DeviceHelpers
{
    internal static class CommonExtensions
    {
        public static Joint ScaleTo(this Joint joint, int width, int height, float skeletonMaxX, float skeletonMaxY)
        {
            Microsoft.Kinect.CameraSpacePoint pos = new CameraSpacePoint()
            {
                X = Scale(width, skeletonMaxX, joint.Position.X),
                Y = Scale(height, skeletonMaxY, -joint.Position.Y),
                Z = joint.Position.Z
            };

            joint.Position = pos;

            return joint;
        }

        public static Joint ScaleTo(this Joint joint, int width, int height)
        {
            return ScaleTo(joint, width, height, 1.0f, 1.0f);
        }

        private static float Scale(int maxPixel, float maxSkeleton, float position)
        {
            float value = ((((maxPixel / maxSkeleton) / 2) * position) + (maxPixel / 2));
            if (value > maxPixel)
                return maxPixel;
            if (value < 0)
                return 0;
            return value;
        }
    }
}
