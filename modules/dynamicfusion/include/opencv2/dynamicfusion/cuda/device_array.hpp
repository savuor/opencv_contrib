#pragma once


#include "device_memory.hpp"

#include <vector>

namespace cv
{
    namespace kfusion
    {
        namespace cuda
        {
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /** \brief @b DeviceArray class
              *
              * \note Typed container for GPU memory with reference counting.
              *
              * \author Anatoly Baksheev
              */
            template<class T>
            class  DeviceArray : public DeviceMemory
        {
            public:
            /** \brief Element type. */
            typedef T type;

            /** \brief Element size. */
            enum { elem_size = sizeof(T) };

            /** \brief Empty constructor. */
            DeviceArray() {}

            /** \brief Allocates internal buffer in GPU memory
              * \param size_t: number of elements to allocate
              * */
            DeviceArray(size_t size) : DeviceMemory(size * elem_size) {}

            /** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
              * \param ptr: pointer to buffer
              * \param size: element number
              * */
            DeviceArray(T *ptr, size_t size) : DeviceMemory(ptr, size * elem_size) {}

            /** \brief Copy constructor. Just increments reference counter. */
            DeviceArray(const DeviceArray& other) : DeviceMemory(other) {}

            /** \brief Assigment operator. Just increments reference counter. */
            DeviceArray& operator=(const DeviceArray& other)
            {
                DeviceMemory::operator=(other);
                return *this;
            }

            /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
              * \param size: elemens number
              * */
            void create(size_t size) { DeviceMemory::create(size * elem_size); }

            /** \brief Decrements reference counter and releases internal buffer if needed. */
            void release() { DeviceMemory::release(); }

            /** \brief Performs data copying. If destination size differs it will be reallocated.
              * \param other_arg: destination container
              * */
            void copyTo(DeviceArray& other) const { DeviceMemory::copyTo(other); }

            /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
              * \param host_ptr_arg: pointer to buffer to upload
              * \param size: elemens number
              * */
            void upload(const T *host_ptr, size_t size) { DeviceMemory::upload(host_ptr, size * elem_size); }

            /** \brief Downloads data from internal buffer to CPU memory
              * \param host_ptr_arg: pointer to buffer to download
              * */
            void download(T *host_ptr) const { DeviceMemory::download(host_ptr); }

            /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
              * \param data: host vector to upload from
              * */
            template<class A>
            void upload(const std::vector <T, A>& data) { upload(&data[0], data.size()); }

            /** \brief Downloads data from internal buffer to CPU memory
              * \param data:  host vector to download to
              * */
            template<typename A>
            void download(std::vector<T, A>& data) const
            {
                data.resize(size());
                if (!data.empty()) download(&data[0]);
            }

            /** \brief Performs swap of data pointed with another device array.
              * \param other: device array to swap with
              * */
            void swap(DeviceArray& other_arg) { DeviceMemory::swap(other_arg); }

            /** \brief Returns pointer for internal buffer in GPU memory. */
            T* ptr() { return DeviceMemory::ptr<T>(); }

            /** \brief Returns const pointer for internal buffer in GPU memory. */
            const T* ptr() const { return DeviceMemory::ptr<T>(); }

            //using DeviceMemory::ptr;

            /** \brief Returns pointer for internal buffer in GPU memory. */
            operator T*() { return ptr(); }

            /** \brief Returns const pointer for internal buffer in GPU memory. */
            operator const T*() const { return ptr(); }

            /** \brief Returns size in elements. */
            size_t size() const { return sizeBytes() / elem_size; }
        };


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /** \brief @b DeviceArray2D class
          *
          * \note Typed container for pitched GPU memory with reference counting.
          *
          * \author Anatoly Baksheev
          */
        template<class T>
        class CV_EXPORTS DeviceArray2D : public DeviceMemory2D
    {
        public:
        /** \brief Element type. */
        typedef T type;

        /** \brief Element size. */
        enum { elem_size = sizeof(T) };

        /** \brief Empty constructor. */
        DeviceArray2D() { }

        /** \brief Allocates internal buffer in GPU memory
          * \param rows: number of rows to allocate
          * \param cols: number of elements in each row
          * */
        DeviceArray2D(int rows, int cols) : DeviceMemory2D(rows, cols * elem_size) {}

        /** \brief Initializes with user allocated buffer. Reference counting is disabled in this case.
         * \param rows: number of rows
         * \param cols: number of elements in each row
         * \param data: pointer to buffer
         * \param stepBytes: stride between two consecutive rows in bytes
         * */
        DeviceArray2D(int rows, int cols, void *data, size_t stepBytes) : DeviceMemory2D(rows, cols*elem_size, data, stepBytes) {}

        /** \brief Copy constructor. Just increments reference counter. */
        DeviceArray2D(const DeviceArray2D& other) : DeviceMemory2D(other) { }

        /** \brief Assigment operator. Just increments reference counter. */
        DeviceArray2D& operator = (const DeviceArray2D& other)
        {
            DeviceMemory2D::operator=(other);
            return *this;
        }

        /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
           * \param rows: number of rows to allocate
           * \param cols: number of elements in each row
           * */
        void create(int rows, int cols)
        {
            DeviceMemory2D::create(rows, cols * elem_size);
        }

        /** \brief Decrements reference counter and releases internal buffer if needed. */
        void release() { DeviceMemory2D::release(); }

        /** \brief Performs data copying. If destination size differs it will be reallocated.
          * \param other: destination container
          * */
        void copyTo(DeviceArray2D& other) const { DeviceMemory2D::copyTo(other); }

        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
          * \param host_ptr: pointer to host buffer to upload
          * \param host_step: stride between two consecutive rows in bytes for host buffer
          * \param rows: number of rows to upload
          * \param cols: number of elements in each row
          * */
        void upload(const void *host_ptr, size_t host_step, int rows, int cols)
        {
            DeviceMemory2D::upload(host_ptr, host_step, rows, cols * elem_size);
        }

        /** \brief Downloads data from internal buffer to CPU memory. User is resposible for correct host buffer size.
          * \param host_ptr: pointer to host buffer to download
          * \param host_step: stride between two consecutive rows in bytes for host buffer
          * */
        void download(void *host_ptr, size_t host_step) const
        {
            DeviceMemory2D::download(host_ptr, host_step);
        }

        /** \brief Performs swap of data pointed with another device array.
          * \param other: device array to swap with
          * */
        void swap(DeviceArray2D& other_arg) { DeviceMemory2D::swap(other_arg); }

        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
          * \param data: host vector to upload from
          * \param cols: stride in elements between two consecutive rows for host buffer
          * */
        template<class A>
        void upload(const std::vector <T, A>& data, int cols)
        {
            upload(&data[0], cols * elem_size, data.size() / cols, cols);
        }

        /** \brief Downloads data from internal buffer to CPU memory
           * \param data: host vector to download to
           * \param cols: Output stride in elements between two consecutive rows for host vector.
           * */
        template<class A>
        void download(std::vector<T, A>& data, int &elem_step) const
        {
            elem_step = cols();
            data.resize(cols() * rows());
            if (!data.empty()) download(&data[0], colsBytes());
        }

        /** \brief Returns pointer to given row in internal buffer.
          * \param y_arg: row index
          * */
        T* ptr(int y = 0) { return DeviceMemory2D::ptr<T>(y); }

        /** \brief Returns const pointer to given row in internal buffer.
          * \param y_arg: row index
          * */        
        const T* ptr(int y = 0) const { return DeviceMemory2D::ptr<T>(y); }

        //using DeviceMemory2D::ptr;

        /** \brief Returns pointer for internal buffer in GPU memory. */
        operator T*() { return ptr(); }

        /** \brief Returns const pointer for internal buffer in GPU memory. */
        operator const T*() const { return ptr(); }

        /** \brief Returns number of elements in each row. */
        int cols() const { return DeviceMemory2D::colsBytes() / elem_size; }

        /** \brief Returns number of rows. */
        int rows() const { return DeviceMemory2D::rows(); }

        /** \brief Returns step in elements. */
        size_t elem_step() const { return DeviceMemory2D::step() / elem_size; }
    };
};

namespace device
{
    using cv::kfusion::cuda::DeviceArray;
    using cv::kfusion::cuda::DeviceArray2D;
}
}
}
