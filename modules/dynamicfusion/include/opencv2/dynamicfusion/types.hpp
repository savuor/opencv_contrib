#pragma once

#include "device_array.hpp"
#include <opencv2/viz/vizcore.hpp>
#include <iosfwd>
#include <iostream>

struct CUevent_st;

namespace cv
{
    namespace kfusion
    {
        typedef cv::Matx33f Mat3f;
        typedef cv::Matx44f Mat4f;
        typedef cv::Vec3f Vec3f;
        typedef cv::Vec4f Vec4f;
        typedef cv::Vec3i Vec3i;
        typedef cv::Affine3f Affine3f;

        struct  Intr
        {
            float fx, fy, cx, cy;

            Intr () { }
            Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) { }
            Intr operator()(int level_index) const
            {
                int div = 1 << level_index;
                return (Intr (fx / div, fy / div, cx / div, cy / div));
            }
        };

        std::ostream& operator << (std::ostream& os, const Intr& intr)
        {
            return os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
        }

        struct Point
        {
            union
            {
                float data[4];
                struct { float x, y, z; };
            };
        };

        typedef Point Normal;

        struct RGB
        {
            union
            {
                struct { unsigned char b, g, r; };
                int bgra;
            };
        };

        struct PixelRGB
        {
            unsigned char r, g, b;
        };

        namespace cuda
        {
            typedef cuda::DeviceMemory CudaData;
            typedef cuda::DeviceArray2D<unsigned short> Depth;
            typedef cuda::DeviceArray2D<unsigned short> Dists;
            typedef cuda::DeviceArray2D<RGB> Image;
            typedef cuda::DeviceArray2D<Normal> Normals;
            typedef cuda::DeviceArray2D<Point> Cloud;

            struct Frame
            {
                bool use_points;

                std::vector<Depth> depth_pyr;
                std::vector<Cloud> points_pyr;
                std::vector<Normals> normals_pyr;
            };
        }

        inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

        struct  ScopeTime
        {
            const char* name;
            double start;
            ScopeTime(const char *name_) : name(name_)
            {
                start = (double)cv::getTickCount();
            }
            ~ScopeTime()
            {
                double time_ms =  ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
                std::cout << "Time(" << name << ") = " << time_ms << "ms" << std::endl;
            }
        };

        struct  SampledScopeTime
        {
        public:
            enum { EACH = 33 };

            SampledScopeTime();
            SampledScopeTime(double& time_ms) : time_ms_(time_ms)
            {
                start = (double)cv::getTickCount();
            }
        private:
            double getTime()
            {
                return ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
            }

            SampledScopeTime(const SampledScopeTime&);
            SampledScopeTime& operator=(const SampledScopeTime&);

            ~SampledScopeTime();

            double& time_ms_;
            double start;
        };

    }
}
