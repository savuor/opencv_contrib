#include "device_array.hpp"
/////////////////////   implementations of DeviceArray ////////////////////////////////////////////
namespace cv
{
    namespace kfusion
    {
        namespace cuda
        {


            template<class T>
            DeviceArray<T>::DeviceArray() {}

            template<class T>
            DeviceArray<T>::DeviceArray(size_t size) : DeviceMemory(size * elem_size) {}

            template<class T>
            DeviceArray<T>::DeviceArray(T *ptr, size_t size) : DeviceMemory(ptr, size * elem_size) {}

            template<class T>
            DeviceArray<T>::DeviceArray(const DeviceArray &other) : DeviceMemory(other) {}

            template<class T>
            DeviceArray<T> &DeviceArray<T>::operator=(const DeviceArray &other) {
                DeviceMemory::operator=(other);
                return *this;
            }

            template<class T>
            void DeviceArray<T>::create(size_t size) { DeviceMemory::create(size * elem_size); }

            template<class T>
            void DeviceArray<T>::release() { DeviceMemory::release(); }

            template<class T>
            void DeviceArray<T>::copyTo(DeviceArray &other) const { DeviceMemory::copyTo(other); }

            template<class T>
            void DeviceArray<T>::upload(const T *host_ptr, size_t size) { DeviceMemory::upload(host_ptr, size * elem_size); }

            template<class T>
            void DeviceArray<T>::download(T *host_ptr) const { DeviceMemory::download(host_ptr); }

            template<class T>
            void DeviceArray<T>::swap(DeviceArray &other_arg) { DeviceMemory::swap(other_arg); }

            template<class T>
            DeviceArray<T>::operator T *() { return ptr(); }

            template<class T>
            DeviceArray<T>::operator const T *() const { return ptr(); }

            template<class T>
            size_t DeviceArray<T>::size() const { return sizeBytes() / elem_size; }

            template<class T>
            T *DeviceArray<T>::ptr() { return DeviceMemory::ptr<T>(); }

            template<class T>
            const T *DeviceArray<T>::ptr() const { return DeviceMemory::ptr<T>(); }

            template<class T>
            template<class A>
            void DeviceArray<T>::upload(const std::vector <T, A> &data) { upload(&data[0], data.size()); }

            template<class T>
            template<class A>
            void DeviceArray<T>::download(std::vector <T, A> &data) const {
                data.resize(size());
                if (!data.empty()) download(&data[0]);
            }
        }
    }
}
