/**
 * Tested on Linux:
 *      - i7 CPU with OpenCL 1.1 (single precision)
 *      - ATI HD4550 with OpenCL 1.1 (single precision)
 *      - nVidia GTX 260 with OpenCL 1.1 (double precision)
 * Tested on Windows 7:
 *      - ATI HD5870 with OpenCL 1.1 (double precision)
 */
#ifndef _OCLKERNEL_HPP_
#define _OCLKERNEL_HPP_

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <CL/cl.hpp>

#include "precision.h"



/**
 * A class to handle OpenCL kernels: source files and binaries.-
 */
class OCLKernel
{
    public:
        /**
         * Constructor
         */
        OCLKernel (const char* filename) : context_ptr(0), queue_ptr(0),
                                           program_ptr(0), kernel_ptr(0),
                                           m_size(0), m_source(0), 
                                           verbose(true), functor_active(false)
        {
            std::ifstream myfile (filename, 
                                  std::ios::in | std::ios::binary | std::ios::ate);
            if (myfile.is_open ( ))
            {
                this->m_size = size_t (myfile.tellg ( )) + 1;
                this->m_source = new char[m_size];
                myfile.seekg (0, std::ios::beg);
                myfile.read (m_source, m_size-1);
                myfile.close ( );
                this->m_source[this->m_size-1] = '\0';
            }
        }
 
        /**
         * Destructor
         */
        virtual ~OCLKernel ( )
        {
            if (this->m_source)
                delete [] this->m_source;
            if (this->kernel_ptr)
                delete this->kernel_ptr;
            if (this->program_ptr)
                delete this->program_ptr;
            if (this->queue_ptr)
                delete this->queue_ptr;
            if (this->context_ptr)
                delete this->context_ptr;
        }

        /**
         * Initilizes the OpenCL platform before kernel execution
         */
        void init (bool verbose=true,
                   bool cpu_only=false)
        {
            this->verbose = verbose;
            try 
            {
                // the platform(s) and their related info
                std::vector<cl::Platform> platforms;
                std::vector<cl::Platform>::iterator itp;
                
                if (this->devices.size ( ) > 0)
                    this->devices.clear ( );
                std::vector<cl::Device>::iterator itd;

                cl::Platform::get (&platforms);

                for (itp = platforms.begin ( ); itp < platforms.end ( ); itp ++)
                {
                    if (this->verbose)
                    {
                        std::string buff;
                        itp->getInfo (CL_PLATFORM_NAME, &buff);
                        std::cout << ":: Platform :: " << buff << std::endl;
                        itp->getInfo (CL_PLATFORM_VENDOR, &buff);
                        std::cout << ":: Vendor :: " << buff << std::endl;
                        itp->getInfo (CL_PLATFORM_VERSION, &buff);
                        std::cout << ":: Version :: " << buff << std::endl;
                        itp->getInfo (CL_PLATFORM_PROFILE, &buff);
                        std::cout << ":: Profile :: " << buff << std::endl;
                    }

                    // query GPU devices in this platform
                    int devn = 0;
                    try
                    {
                        itp->getDevices (CL_DEVICE_TYPE_GPU, &(this->devices));
                    }
                    catch (cl::Error error) 
                    {
                        this->devices.clear ( );
                    }

                    // switch to CPU if there is no GPU 
                    // available or CPU has been forced
                    if (cpu_only || this->devices.size ( ) == 0)
                    {
                        std::cerr << "::: WARNING switching over to CPU." << std::endl;
                        itp->getDevices (CL_DEVICE_TYPE_CPU, &(this->devices));
                        assert (this->devices.size ( ) > 0);
                    }
                    if (this->verbose)
                    {
                        for (itd = this->devices.begin ( ); itd < this->devices.end ( ); itd ++)
                        {
                            std::string buff2;
                            itd->getInfo (CL_DEVICE_NAME, &buff2);
                            std::cout << "\t|| Device " << devn << " || " << buff2 << std::endl;
                            itd->getInfo (CL_DEVICE_VENDOR, &buff2);
                            std::cout << "\t|| Vendor " << devn << " || " << buff2 << std::endl;
                            itd->getInfo (CL_DEVICE_MAX_WORK_GROUP_SIZE, &(this->max_wgroup_size));
                            std::cout << "\t|| Maximum threads per block || "
                                      << this->max_wgroup_size << std::endl;
                            itd->getInfo (CL_DEVICE_LOCAL_MEM_SIZE, &(this->local_mem_size));
                            std::cout << "\t|| Local memory size || "
                                      << this->local_mem_size << std::endl;
                            devn ++;
                        }
                    }
                }
                // delete any previous references
                if (this->context_ptr)
                    delete this->context_ptr;
                if (this->queue_ptr)
                    delete this->queue_ptr;

                // create a context and a command queue
                this->context_ptr = new cl::Context (this->devices);
                this->context = *(this->context_ptr);
                this->queue_ptr = new cl::CommandQueue (this->context,
                                                        this->devices[0]);
                this->queue = *(this->queue_ptr);

                if (this->verbose)
                {
                    std::string buff;
                    itd = devices.begin ( );
                    itd->getInfo (CL_DEVICE_NAME, &buff);
                    std::cout << ":: OpenCL Context initialized for device -- "
                              << buff << " -- " << std::endl;
                }
            } 
            catch (cl::Error &error)
            {
                std::cerr << "::: ERROR: initialization failed!" << std::endl;
                std::cerr << "::: ERROR: " << error.what ( ) 
                          << "(" << error.err ( ) << ")"
                          << std::endl;
            }
        }

        /**
         * Transfers the data pointed by 'device_data' from the device,
         * to the address pointed by 'host_data' on the host.
         */
        void read_buffer (const cl::Buffer &device_data,
                          void *host_data, 
                          const size_t data_size)
        {
            cl_int error = this->queue.enqueueReadBuffer (device_data,
                                                          CL_TRUE, 
                                                          0, 
                                                          data_size, 
                                                          host_data);
            if (error != CL_SUCCESS)
            {
                std::cerr << "::: ERROR reading data from device" << std::endl;
            }
        }

        /**
         * Transfers the data pointed by 'host_data' on the host,
         * to the address pointed by 'device_data' at the device.
         */
        void write_buffer (const cl::Buffer &device_data,
                           const void *host_data, 
                           const size_t data_size)
        {
            cl_int error = this->queue.enqueueWriteBuffer (device_data,
                                                           CL_TRUE, 
                                                           0, 
                                                           data_size, 
                                                           host_data);
            if (error != CL_SUCCESS)
            {
                std::cerr << "::: ERROR writing data to device" << std::endl;
            }
        }

        /**
         * Compiles the kernel code passed as a constructor parameter.
         */
        void build (const char * options=NULL)
        {
            // compile only if there is a valid kernel
            if (this->m_size > 0)
            {
                try
                {
                    if (this->verbose)
                    {
                        std::cout << ":: Building kernel binary ... " << std::endl;
                        std::cout << "\t|| Options ||\t" << options << std::endl;
                    }
                    if (this->program_ptr)
                        delete this->program_ptr;

                    cl::Program::Sources clsource (1, this->get_source_size_pair ( ));
                    this->program_ptr = new cl::Program (this->context, clsource);
                    this->program_ptr->build (this->devices, options);
                    this->program = *(this->program_ptr);

                    // release compiler resources
                    clUnloadCompiler ( );
                }
                catch (cl::Error &error)
                {
                    std::cerr << "::: ERROR: kernel compilation failed!" << std::endl;
                    std::cerr << "::: ERROR: " << error.what ( )
                              << "(" << error.err ( ) << ")" << std::endl;
                }
            }
            else
            {
                std::cerr << "::: ERROR: no kernel source file provided!" << std::endl;
            }
        }


        /**
         * Activates one kernel function from the compiled binary kernels
         * received as constructor parameters.
         * The activated kernel is implicitly used in later function calls.
         */
        void activate_kernel (const char *kernel_name)
        {
            cl_int error;

            if (this->verbose)
            {
                std::cout << ":: Activating kernel <"
                          << kernel_name 
                          << "> ... ";
                std::cout.flush ( );
            }
            if (this->kernel_ptr)
                delete this->kernel_ptr;

            this->kernel_ptr = new cl::Kernel (this->program, kernel_name, &error);

            if (error == CL_SUCCESS)
            {
                this->kernel = *(this->kernel_ptr);
                if (this->verbose)
                    std::cout << "done!" << std::endl;
            }
            else
            {
                this->kernel_ptr = 0;
                std::cerr << "::: ERROR kernel activation failed ("
                          << error << ")"
                          << std::endl;
            }
        }

        
        /**
         * Sets a N-dimensional execution range to the activated kernel.
         */
        void set_range (const int dimension,
                        const size_t global_sizes [], 
                        const size_t local_sizes [],
                        const size_t offsets [] = NULL)
        {
            const size_t zero_offsets [] = {0, 0, 0};

            if (this->kernel_ptr)
            {
                // if the offsets are not given, take zero as default
                if (offsets == NULL)
                    offsets = zero_offsets;

                // create range objects based on the given dimension
                switch (dimension)
                {
                    case (1):
                        this->global = cl::NDRange (global_sizes[0]);
                        this->local = cl::NDRange (local_sizes[0]);
                        this->offset = cl::NDRange (offsets[0]);
                        break;

                    case (2):
                        this->global = cl::NDRange (global_sizes[0], global_sizes[1]);
                        this->local = cl::NDRange (local_sizes[0], local_sizes[1]);
                        this->offset = cl::NDRange (offsets[0], offsets[1]);
                        break;

                    case (3):
                        this->global = cl::NDRange (global_sizes[0], global_sizes[1], global_sizes[2]);
                        this->local = cl::NDRange (local_sizes[0], local_sizes[1], local_sizes[2]);
                        this->offset = cl::NDRange (offsets[0], offsets[1], offsets[2]);
                        break;
                }
                // check that the execution range is valid
                unsigned int i, wgroup_size = 1, total_threads = 1;

                if ((this->global.dimensions ( ) > 0) &&
                    (this->global.dimensions ( ) == this->local.dimensions ( )) &&
                    (this->local.dimensions ( ) == this->offset.dimensions ( )))
                {
                    for (i = 0; i < this->global.dimensions ( ); i ++)
                    {
                        if ((global_sizes[i] % local_sizes[i]) != 0)
                        {
                            std::cerr << "::: ERROR local size must divide global size" << std::endl;
                            return;
                        }
                        // calculate the total number of threads per block
                        wgroup_size *= local_sizes[i];
                        // calculate the total number of threads
                        total_threads *= global_sizes[i];
                    }
                    if (wgroup_size > this->max_wgroup_size)
                    {
                        std::cerr << "::: ERROR local work group size exceeds hardware limit ";
                        std::cerr << "(" << this->max_wgroup_size << ")" << std::endl;
                        return;
                    }
                    if (total_threads < wgroup_size)
                    {
                        std::cerr << "::: ERROR global size should be greater ";
                        std::cerr << "or equal than local size" << std::endl;
                        return;
                    }
                }
                else
                {
                     std::cerr << "::: ERROR kernel range is invalid" << std::endl;
                     return;
                }
                if (this->verbose)
                {
                    std::cout << ":: " 
                              << this->global.dimensions ( ) 
                              << "D kernel execution range set to"
                              << std::endl;
                    std::cout << "\tGlobal:";
                    for (i = 0; i < this->global.dimensions ( ); i ++)
                    {
                        std::cout << "\t" << this->global[i];
                    }
                    std::cout << "\n\tLocal:";
                    for (i = 0; i < this->local.dimensions ( ); i ++)
                    {
                        std::cout << "\t" << this->local[i] << " ";
                    }
                    std::cout << "\n\tOffset:";
                    for (i = 0; i < this->offset.dimensions ( ); i ++)
                    {
                        std::cout << "\t" << this->offset[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            else
            {
                std::cerr << "::: ERROR activate a kernel before calling 'set_range(...)'" << std::endl;
            }
        }


        /**
         * Sets a 1-dimensional execution range to the activated kernel.
         */
        void set_1D_range (const size_t global_sizes [], 
                           const size_t local_sizes [],
                           const size_t offsets [] = NULL)
        {
            this->set_range (1, 
                             global_sizes, 
                             local_sizes, 
                             offsets);
        }

        /**
         * Sets a 2-dimensional execution range to the activated kernel.
         */
        void set_2D_range (const size_t global_sizes [], 
                           const size_t local_sizes [],
                           const size_t offsets [] = NULL)
        {
            this->set_range (2,
                             global_sizes, 
                             local_sizes, 
                             offsets);
        }

        /**
         * Sets a 3-dimensional execution range to the activated kernel.
         */
        void set_3D_range (const size_t global_sizes [], 
                           const size_t local_sizes [],
                           const size_t offsets [] = NULL)
        {
            this->set_range (3,
                             global_sizes, 
                             local_sizes, 
                             offsets);
        }


        /**
         * Returns a pointer to an array of 1, 2 or 3
         * elements of the global range.-
         */
        const size_t* get_global_range ( )
        {
        	return &(this->global[0]);
        }


        /**
         * Allocates the specified 'size' bytes of local memory.
         * The 'index' parameter indicates the argument index marked
         * as __local in the kernel function prototype. 
         */
        void set_local (const unsigned int index,
                        const size_t size)
        {
            if (this->verbose)
            {
                std::cout << ":: Allocating " << size
                          << " bytes of local memory ..."
                          << std::endl;
            }
            if (this->local_mem_size > size)
            {
                cl::LocalSpaceArg local_mem = cl::__local (size);
                this->set_arg (index, local_mem);
            }
            else
            {
                std::cerr << "::: ERROR cannot allocate "
                          << size << " bytes of local memory. "
                          << "Hardware limit is " 
                          << this->local_mem_size << " bytes."
                          << std::endl;
            }
        }


        /**
         * Sets the argument value for a specific kernel parameter.
         * The 'index' parameter indicates the argument index. 
         * Arguments to the kernel are referred by indices that go 
         * from 0 for the leftmost argument to n - 1, where n is the
         * total number of arguments declared by a kernel.
         */
        template <typename T>
        void set_arg (unsigned int index, T value)
        {
            if (this->verbose)
            {
                std::cout << "\tSetting parameter " << index
                          << " with size " << sizeof(T)
                          << std::endl;
            }
            cl_int error = this->kernel.setArg (index, value);

            if (error != CL_SUCCESS)
            {
                std::cerr << "::: ERROR setting " << index 
                          << " kernel value parameter!" << std::endl;
            }
        }


        /**
         * Runs the activated kernel.
         * It waits for it to finish execution 
         * based on the value of 'wait'.
         */
         void run (bool wait=false)
         {
            if (this->kernel_ptr)
            {
                if ((this->global.dimensions ( ) > 0) &&
                    (this->global.dimensions ( ) == this->local.dimensions ( )) &&
                    (this->local.dimensions ( ) == this->offset.dimensions ( )))
                {
                    try
                    {
                        if (this->verbose)
                            std::cout << ":: Kernel execution started ... ";
                        
                        // run the kernel with the given execution range
                        cl_int error = this->queue.enqueueNDRangeKernel (this->kernel,
                                                                         this->offset,
                                                                         this->global,
                                                                         this->local);
                        if (error == CL_SUCCESS)
                        {
                            // wait for the kernel to finish?
                            if (wait)
                            {
                                this->queue.finish ( );
                                if (this->verbose)
                                    std::cout << "done!";
                            }
                            if (this->verbose)
                                std::cout << std::endl;
                        }
                        else
                        {
                            std::cerr << "::: ERROR kernel execution failed"
                                      << " (" << error << ")" << std::endl;
                        }
                    }
                    catch (cl::Error &error)
                    {
                        std::cerr << "::: ERROR kernel execution failed!" << std::endl;
                        std::cerr << "::: ERROR " << error.what ( )
                                  << "(" << error.err ( ) << ")" << std::endl;
                    }
                }
                else
                {
                    std::cerr << "::: ERROR a valid kernel range has to "
                              << "be defined before calling 'run_and_wait(...)'" << std::endl;
                }
            }
            else
            {
                std::cerr << "::: ERROR: a kernel has to be activated "
                          << "before calling 'run_and_wait(...)'" << std::endl;
            }
        }

        /**
         * Runs the activated kernel and waits for it to finish execution.
         */
        void run_and_wait ( )
        {
            this->run (true);
        }
        
        const cl::Context& get_context ( )
        {
            return this->context;
        }

        const cl::Device& get_device ( )
        {
            return this->devices[0];
        }

        const char* get_source ( )
        {
            return this->m_source;
        }
 
        size_t get_source_size ( )
        {
            return this->m_size;
        }

 
        private:
            cl::Context *context_ptr;
            cl::Context context;
            cl::CommandQueue *queue_ptr;
            cl::CommandQueue queue;
            std::vector<cl::Device> devices;
            cl::Program *program_ptr;
            cl::Program program;
            cl::Kernel *kernel_ptr;
            cl::Kernel kernel;
            cl::KernelFunctor functor;
            size_t m_size;
            char *m_source;
            bool verbose;
            bool functor_active;
            size_t max_wgroup_size;
            cl_ulong local_mem_size;
            cl::NDRange global;
            cl::NDRange local;
            cl::NDRange offset;


            std::pair<char*, size_t> get_source_size_pair ( )
            {
                return std::make_pair (m_source, m_size);
            }
};

#endif

