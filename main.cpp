#include "oclkernel.hpp"


/**
 * Program entry point
 */
int main (int argc, char** argv)
{
    unsigned int i, j;
    // Number of elements and size of the matrix used as test data
    const unsigned int wh = 16;
    const unsigned int ht = 16;
    const unsigned int nelem = wh*ht;
    size_t memSize = sizeof(real)*nelem;
   
    //
    // Here we create and initialize the matrix used for testing.
    // REMEMBER: Since this is C++, always create arrays with 'new'.
    // NEVER create C-style arrays, i.e. float data [1024];
    //
    // For an explanation, see http://c-faq.com/aryptr/aryptr2.html
    //
    real *data = new real [nelem];
    real *results = new real [nelem];

    for (i = 0; i < nelem; i++)
    {
        data[i] = rand ( ) / (real)RAND_MAX;
    }

    try 
    {
        // Create a new kernel object by passing the
        // source file path to the constructor
        OCLKernel kernel ("other_square.cl");

        // Initialize the OpenCL backend
        kernel.init ( );

        // Get a pointer to the initialized OpenCL context
        cl::Context ctx = kernel.get_context ( );

        // Declare memory on the device, used as kernel parameters
        cl::Buffer input (ctx, CL_MEM_READ_ONLY, memSize);
        cl::Buffer output (ctx, CL_MEM_WRITE_ONLY, memSize);
 
        // Send data to the device
        kernel.write_buffer (input, data, memSize);

        // Compile the kernel source file, passing the
        // include path to the compiler and (possible) constants
        std::string build_options = "-D_MY_CONSTANT_=1 -I.";
        kernel.build (build_options.c_str ( ));

        // Activate a kernel function
        kernel.activate_kernel ("square");

        // Define the kernel execution range; 
        // this example is for a 2-dimensional range ...
        size_t global_sizes [] = {wh, ht};
        size_t local_sizes [] = {wh, ht};
        
        kernel.set_2D_range (global_sizes, local_sizes);

        // ... offsets are optional, default is zero, e.g.
        //
        //size_t offsets [] = {5, 5};
        //kernel.set_2D_range (global_sizes, local_sizes, offsets);

        //
        // An example for a 3-dimensional execution range:
        //
        //size_t global_sizes [] = {16, 16, 16};
        //size_t local_sizes [] = {8, 8, 8};
        //size_t offsets [] = {0, 0 , 0};
        //
        //kernel.set_3D_range (global_sizes, local_sizes, offsets);
        //

        // Set kernel parameters
        kernel.set_arg (0, input);
        kernel.set_arg (1, output);

        // Run the kernel function, waiting for it to finish
        kernel.run_and_wait ( );

        // Enqueue kernel execution and go on (don't wait)
        //kernel.run ( );
    
        // Transfer the results back from the device
        kernel.read_buffer (output, results, memSize);
    } 
    catch (cl::Error &error)
    {
        std::cerr << "::: ERROR "
                  << error.what ( ) 
                  << "(" << error.err ( ) << ")"
                  << std::endl;
    }
    
    std::cout << "Testing results ..." << std::endl;
    unsigned int correct = 0;
    for (i = 0; i < wh; ++i)
    {
        for (j = 0; j < ht; ++j)
        {
            unsigned int elem = i + (j * wh);
            if (results[elem] == data[elem]*data[elem])
                ++correct;
            else
                std::cout << i << ", " << j << '\t' 
                          << results[elem] << '\t' 
                          << data[elem]*data[elem] 
                          << std::endl;
        }
    }
    std::cout << "Computed " << correct << "/" << nelem;
    std::cout << " correct values." << std::endl;

    // Free allocated resources
    delete [] results;
    delete [] data;

    return 0;
}

