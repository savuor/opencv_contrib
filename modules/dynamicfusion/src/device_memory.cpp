#include "device_memory.hpp"
#include "safe_call.hpp"
#include <cassert>
#include <iostream>
#include <cstdlib>
using namespace cv;
void kfusion::cuda::error(const char *error_string, const char *file, const int line, const char * /* func */)
{
    std::cout << "KinFu2 error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}



////////////////////////    DeviceArray    /////////////////////////////
    
kfusion::cuda::DeviceMemory::DeviceMemory() : data_(0), sizeBytes_(0), refcount_(0) {}
kfusion::cuda::DeviceMemory::DeviceMemory(void *ptr_arg, size_t sizeBytes_arg) : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(0){}
kfusion::cuda::DeviceMemory::DeviceMemory(size_t sizeBtes_arg)  : data_(0), sizeBytes_(0), refcount_(0) { create(sizeBtes_arg); }
kfusion::cuda::DeviceMemory::~DeviceMemory() { release(); }

kfusion::cuda::DeviceMemory::DeviceMemory(const DeviceMemory& other_arg)
    : data_(other_arg.data_), sizeBytes_(other_arg.sizeBytes_), refcount_(other_arg.refcount_)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

kfusion::cuda::DeviceMemory& kfusion::cuda::DeviceMemory::operator = (const kfusion::cuda::DeviceMemory& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        data_      = other_arg.data_;
        sizeBytes_ = other_arg.sizeBytes_;                
        refcount_  = other_arg.refcount_;
    }
    return *this;
}

void kfusion::cuda::DeviceMemory::create(size_t /* sizeBytes_arg */)
{
    throw "Not implemented";
    /*
    if (sizeBytes_arg == sizeBytes_)
        return;
            
    if( sizeBytes_arg > 0)
    {        
        if( data_ )
            release();

        sizeBytes_ = sizeBytes_arg;
                        
        cudaSafeCall( cudaMalloc(&data_, sizeBytes_) );
        
        //refcount_ = (int*)cv::fastMalloc(sizeof(*refcount_));
        refcount_ = new int;
        *refcount_ = 1;
    }
    */
}

void kfusion::cuda::DeviceMemory::copyTo(DeviceMemory& other) const
{
    throw "Not implemented";
    /*
    if (empty())
        other.release();
    else
    {    
        other.create(sizeBytes_);    
        cudaSafeCall( cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice) );
    }
    */
}

void kfusion::cuda::DeviceMemory::release()
{
    throw "Not implemented";
    /*
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        //cv::fastFree(refcount);
        delete refcount_;
        cudaSafeCall( cudaFree(data_) );
    }
    data_ = 0;
    sizeBytes_ = 0;
    refcount_ = 0;
    */
}

void kfusion::cuda::DeviceMemory::upload(const void *host_ptr_arg, size_t sizeBytes_arg)
{
    throw "Not implemented";
    /*
    create(sizeBytes_arg);
    cudaSafeCall( cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice) );
    */
}

void kfusion::cuda::DeviceMemory::download(void *host_ptr_arg) const
{    
    /*
    cudaSafeCall( cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost) );
    */
}          

void kfusion::cuda::DeviceMemory::swap(DeviceMemory& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(sizeBytes_, other_arg.sizeBytes_);
    std::swap(refcount_, other_arg.refcount_);
}

bool kfusion::cuda::DeviceMemory::empty() const { return !data_; }
size_t kfusion::cuda::DeviceMemory::sizeBytes() const { return sizeBytes_; }



