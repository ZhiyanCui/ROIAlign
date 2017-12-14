#include <stdio.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "matrix.h"
#include <assert.h>
#include <limits>
#include <cassert>

#include <cstring>
#include <cmath>
#include <iostream>

//the global parm
const int sampleNum=2;
const int filterDem=4;

//#define debug

struct Bounds 
{
    int offset; 	
    float hstart, hend, wstart, wend ;
    bool isEmpty ;
} ;

    __device__ __forceinline__ static  Bounds getBounds
(int outputIndex,
 int height, int width, int numChannels, int size,
 const float* rois, int numROIs,
 int  subd,float transform)
{
    Bounds b ;
    int ph = outputIndex ;
    int pw = ph / subd ;
    int pc = pw / subd;
    int pr = pc / numChannels ;

    ph %= subd ;
    pw %= subd ;
    pc %= numChannels ;

    rois += 5 * pr ;

    // Apply sacle and offset to each ROI coordinate.
    float u1_ = rois[1] ;
    float v1_ = rois[2] ;
    float u2_ = rois[3] ;
    float v2_ = rois[4] ;

    float u1 = transform * (u1_ - 1) + 1;
    float v1 = transform * (v1_ - 1) + 1;
    float u2 = transform * (u2_ - 1) + 1;
    float v2 = transform * (v2_ - 1) + 1;

    int roi_image   = (int) rois[0];
    float roi_start_h = v1 - 1 ;
    float roi_start_w = u1 - 1 ;
    float roi_end_h   = v2 - 1 ;
    float roi_end_w   = u2 - 1 ;
    float roi_height  =max(roi_end_h - roi_start_h + 1.0, 1.0) ;
    float roi_width   = max(roi_end_w - roi_start_w + 1.0, 1.0) ;

    float bin_size_h = (float)roi_height / subd ;
    float bin_size_w = (float)roi_width / subd ;

    roi_image = min(max(roi_image - 1,0), (int)size - 1) ;
    b.offset = (roi_image * numChannels + pc) * (width * height) ;

    b.wstart = (float)(((float)pw) * bin_size_w) ;
    b.wend = (float)(((float)(pw + 1)) * bin_size_w) ;
    b.wstart = min(max(b.wstart + roi_start_w,(float) 0.0), (float)width) ;
    b.wend =min(max(b.wend + roi_start_w,(float)0.0), (float)width) ;

    b.hstart = (float)(((float)ph) * bin_size_h) ;
    b.hend = (float)((float)(ph + 1) * bin_size_h) ;
    b.hstart = min(max(b.hstart + roi_start_h, (float)0.0), (float)height) ;
    b.hend = min(max(b.hend + roi_start_h, (float)0.0), (float)height) ;


    b.isEmpty = (b.hend <= b.hstart) || (b.wend <= b.wstart) ;

    return b ;
}



    void __global__ roialign_max_froward
(float* output,
 const float* data, int height, int width, int numChannels, int size,
 const float* rois, int numROIs,
 int subd,float transform)
{
    int outputIndex = threadIdx.x + blockIdx.x * blockDim.x ;
    int outputVolume = subd * subd * numChannels * numROIs ;

    //	if (outputIndex < outputVolume) 
    if (outputIndex < outputVolume)

    {

	Bounds b = getBounds(outputIndex,
		height,width,numChannels,size,
		rois,numROIs,
		subd,transform) ;

	if (! b.isEmpty)
	{
	    data += b.offset ;
	    // Define an empty pooling region to be zero
	    float maxval =  -FLT_MAX;
	    float maxidx_x = 0.0;
	    float maxidx_y = 0.0;
	    float w_bin=(b.wend-b.wstart)/(float)(sampleNum+1);
	    float h_bin=(b.hend-b.hstart)/(float)(sampleNum+1);

	    int iter_x=0;
	    for (float w = b.wstart+w_bin; ;w=w+w_bin ) 
	    {
		iter_x++;
		int iter_y=0;
		if(iter_x > sampleNum)
		{
		    break;
		}
		for (float h = b.hstart+h_bin; ;h=h+h_bin) 
		{
		    iter_y++;
		    if(iter_y > sampleNum)
		    {
			break;
		    }
		    // Selecting four regular locations for bilinear interpolation
		    int x_left = floor(w);
		    int x_right = ceil(w);
		    int y_top = floor(h);
		    int y_bottom = ceil(h);

		    int top_left_index = x_left * height + y_top;
		    int bottom_left_index = x_left * height + y_bottom;
		    int top_right_index = x_right * height + y_top;
		    int bottom_right_index = x_right * height + y_bottom;

		    bool is_top_left_in = x_left >= 0 && x_left <= width - 1
			&& y_top >= 0 && y_top <= height - 1;
		    bool is_top_right_in = x_right >= 0 && x_right <= width - 1
			&& y_top >= 0 && y_top <= height - 1;
		    bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
			&& y_bottom >= 0 && y_bottom <= height - 1;
		    bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
			&& y_bottom >= 0 && y_bottom <= height - 1;

		    float val = 0.0;
		    if (is_top_left_in)
		    {
			val += (1 - (w - x_left)) * (1 - (h-y_top)) * data[top_left_index]; 
		    }
		    if (is_top_right_in)
		    {
			val += (1 - (x_right-w)) * (1 - (h-y_top)) * data[top_right_index];
		    }					
		    if (is_bottom_left_in)
		    {
			val += (1 - (w-x_left)) * (1 - (y_bottom-h)) * data[bottom_left_index]; 
		    }
		    if (is_bottom_right_in)
		    {
			val += (1-(x_right-w)) * (1-(y_bottom-h)) * data[bottom_right_index]; 
		    }

		    if (val > maxval) 
		    {
			maxval = val;
			maxidx_x = w;
			maxidx_y = h;

		    }
			
		}
	    }
	    output[outputIndex] = maxval ;
	} 
	else 
	{
	    output[outputIndex] = 0 ;
	}
    }

}


    void __global__ roialign_max_backward
(const float* derData,
 const float* data, int height, int width, int numChannels, int size,
 const float* rois, int numROIs,
 float* derOutput, int subd,float transform)
{
    int outputIndex = threadIdx.x + blockIdx.x * blockDim.x ;
    int outputVolume = subd * subd * numChannels * numROIs ;
    //	if (outputIndex < outputVolume) 
    if (outputIndex < outputVolume)
    {

	Bounds b = getBounds(outputIndex,
		height,width,numChannels,size,
		rois,numROIs,
		subd,transform) ;

	if (! b.isEmpty)
	{
	    data += b.offset ;
	    derData += b.offset ;
	    // Define an empty pooling region to be zero
	    float maxval =  -FLT_MAX;

	    int index_left_top;
	    int index_right_top;
	    int index_left_bottom;
	    int index_right_bottom ;


	    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
	    float maxidx_x = 0.0;
	    float maxidx_y = 0.0; 
	    float w_bin=(b.wend-b.wstart) / (float)(sampleNum+1);
	    float h_bin=(b.hend-b.hstart) / (float)(sampleNum+1);

	    int iter_x=0;
		
		// Selecting 2*2 regular locations for bilinear interpolation

	    for (float w = b.wstart + w_bin; ;w = w + w_bin ) 
	    {
		iter_x++;
		int iter_y = 0;
		if(iter_x > sampleNum)
		{
		    break;
		}
		for (float h = b.hstart + h_bin; ;h = h + h_bin) 
		{
		    iter_y++;
		    if(iter_y > sampleNum)
		    {
			break;
		    }
		    
		    int x_left = floor(w);
		    int x_right = ceil(w);
		    int y_top = floor(h);
		    int y_bottom = ceil(h);

		    int top_left_index = x_left * height + y_top;
		    int bottom_left_index = x_left * height + y_bottom;
		    int top_right_index = x_right * height + y_top;
		    int bottom_right_index = x_right * height + y_bottom;

		    bool is_top_left_in = x_left >= 0 && x_left <= width - 1
			&& y_top >= 0 && y_top <= height - 1;
		    bool is_top_right_in = x_right >= 0 && x_right <= width - 1
			&& y_top >= 0 && y_top <= height - 1;
		    bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
			&& y_bottom >= 0 && y_bottom <= height - 1;
		    bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
			&& y_bottom >= 0 && y_bottom <= height - 1;

		    float val = 0.0;
		    if (is_top_left_in)
		    {
			val += (1 - (w - x_left)) * (1 - (h-y_top)) * data[top_left_index]; 
		    }
		    if (is_top_right_in)
		    {
			val += (1 - (x_right-w)) * (1 - (h-y_top)) * data[top_right_index];
		    }					
		    if (is_bottom_left_in)
		    {
			val += (1 - (w-x_left)) * (1 - (y_bottom-h)) * data[bottom_left_index]; 
		    }
		    if (is_bottom_right_in)
		    {
			val += (1-(x_right-w)) * (1-(y_bottom-h)) * data[bottom_right_index]; 
		    }

		    if (val > maxval) 
		    {
			maxval = val;
			maxidx_x = w;
			maxidx_y = h;

			index_left_top = floor(maxidx_x) * height + floor(maxidx_y);
			index_right_top = ceil(maxidx_x) * height + floor(maxidx_y);
			index_left_bottom = floor(maxidx_x) * height + ceil(maxidx_y);
			index_right_bottom = ceil(maxidx_x) * height + ceil(maxidx_y);
		    }
				
		}
	    }

	    atomicAdd(derOutput + index_left_top, (1-(maxidx_x-floor(maxidx_x))) * (1-(maxidx_y-floor(maxidx_y))) * derOutput[outputIndex]) ;
	    atomicAdd(derOutput + index_left_bottom, (1-(maxidx_x-floor(maxidx_x))) * (1-(ceil(maxidx_y)-maxidx_y)) *derOutput[outputIndex]) ;
	    atomicAdd(derOutput + index_right_top, (1-(ceil(maxidx_x)-maxidx_x)) * (1-(maxidx_y-floor(maxidx_y))) *derOutput[outputIndex]) ;
	    atomicAdd(derOutput + index_right_bottom, (1-(ceil(maxidx_x)-maxidx_x)) * (1-(ceil(maxidx_y)-maxidx_y)) *derOutput[outputIndex]) ;	

	}

    } 

}


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{

    /* Declare all variables.*/
    mxGPUArray const *in;
    mxGPUArray *out;
    mxGPUArray const *rois;

    int subd;
    float transform;

    float const *p_in;
    float const *p_rois; 
    float *p_out;
    float const *p_derin;


    //int N;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    char const * const errInput = "the input is not suitale.";

    /* Choose a reasonably sized number of threads for the block. */
    int threadsPerBlock = 1024;
    int blocksPerGrid;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    if (nrhs == 4) 
    {

	in = mxGPUCreateFromMxArray(prhs[0]);
	rois = mxGPUCreateFromMxArray(prhs[1]);
	subd = mxGetScalar(prhs[2]);	
	transform = mxGetScalar(prhs[3]);
	/*
	 * Verify that in really is a single array before extracting the pointer.
	 */
	if ((mxGPUGetClassID(in) != mxSINGLE_CLASS)||(mxGPUGetClassID(rois) != mxSINGLE_CLASS))
	{
	    mexErrMsgIdAndTxt(errId, errMsg);
	}
	if((int)(mxGPUGetNumberOfDimensions(in)) != 4 && (int)(mxGPUGetNumberOfDimensions(in)) != 3) 
	{
	    mexErrMsgIdAndTxt(errId, errInput);
	}
	if((int)(mxGPUGetNumberOfDimensions(rois)) != 2) 
	{
	    mexErrMsgIdAndTxt(errId, errInput);
	}

	
	p_in = (float const *)(mxGPUGetDataReadOnly(in));
	p_rois = (float const *)(mxGPUGetDataReadOnly(rois));

	
	mwSize const *in_dim = mxGPUGetDimensions(in);
	mwSize const *rois_dim = mxGPUGetDimensions(rois);

	int height = in_dim[0];
	int width = in_dim[1];
	int numChannels = in_dim[2];	
	int numROIs = rois_dim[1];
	int size = 1;
	
	if((int)(mxGPUGetNumberOfDimensions(in)) == 4) 
	{
	    size = in_dim[3];
	}


	mwSize dim[4] = {(mwSize)subd,(mwSize)subd,in_dim[2],rois_dim[1]};


	blocksPerGrid=(subd * subd * numChannels * numROIs + threadsPerBlock -1 ) / threadsPerBlock;


    	//mxGPUGetNumberOfDimensions(in)
	out = mxGPUCreateGPUArray((mwSize)filterDem,
		dim,
		mxGPUGetClassID(in),
		mxGPUGetComplexity(in),
		MX_GPU_DO_NOT_INITIALIZE );
	p_out = (float *)(mxGPUGetData(out));


	/*
	 * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
	 * and it would be possible for the number of elements to be too large for
	 * the grid. For this example we are not guarding against this possibility.
	 */

	roialign_max_froward
	    <<< blocksPerGrid,threadsPerBlock>>>
	    (p_out,
	     p_in, height, width, numChannels, size,
	     p_rois, numROIs,
	     subd,transform);

	/* Wrap the result up as a MATLAB gpuArray for return. */
	plhs[0] = mxGPUCreateMxArrayOnGPU(out);

	/*
	 * The mxGPUArray pointers are host-side structures that refer to device
	 * data. These must be destroyed before leaving the MEX function.
	 */
	mxGPUDestroyGPUArray(in);
	mxGPUDestroyGPUArray(out);
	mxGPUDestroyGPUArray(rois);
    }
    else if(nrhs==5)
    {

	mxGPUArray const *derin;
	in = mxGPUCreateFromMxArray(prhs[0]);
	rois = mxGPUCreateFromMxArray(prhs[1]);
	derin = mxGPUCreateFromMxArray(prhs[2]);
	subd = mxGetScalar(prhs[3]);	
	transform = mxGetScalar(prhs[4]);


	if ((mxGPUGetClassID(in) != mxSINGLE_CLASS)||(mxGPUGetClassID(rois) != mxSINGLE_CLASS)
		||(mxGPUGetClassID(derin) != mxSINGLE_CLASS)			)
	{
	    mexErrMsgIdAndTxt(errId, errMsg);
	}
	if((int)(mxGPUGetNumberOfDimensions(in)) != 4 && (int)(mxGPUGetNumberOfDimensions(in)) != 3) 
	{
	    mexErrMsgIdAndTxt(errId, errInput);
	}
	if((int)(mxGPUGetNumberOfDimensions(rois)) !=2 ) 
	{
	    mexErrMsgIdAndTxt(errId, errInput);
	}

	p_in = (float const *)(mxGPUGetDataReadOnly(in));
	p_rois = (float const *)(mxGPUGetDataReadOnly(rois));
	p_derin = (float const *)(mxGPUGetDataReadOnly(derin));

	mwSize const* in_dim = mxGPUGetDimensions(in);
	mwSize const* rois_dim = mxGPUGetDimensions(rois);

	int height = in_dim[0];
	int width = in_dim[1];
	int numChannels = in_dim[2];
	int numROIs = rois_dim[1];
	int size = 1;
	if((int)(mxGPUGetNumberOfDimensions(in)) == 4) 
	{
	    size = in_dim[3];
	}

	blocksPerGrid = (subd * subd * numChannels * numROIs + threadsPerBlock - 1) / threadsPerBlock;

	out = mxGPUCreateGPUArray((mwSize)filterDem ,
		mxGPUGetDimensions(in),
		mxGPUGetClassID(in),
		mxGPUGetComplexity(in),
		MX_GPU_INITIALIZE_VALUES );
	p_out = (float *)(mxGPUGetData(out));

	roialign_max_backward
	    <<< blocksPerGrid,threadsPerBlock>>>
	    (p_derin, p_in,
	     height, width, numChannels, size,
	     p_rois, numROIs,
	     p_out,
	     subd,transform) ;

	/* Wrap the result up as a MATLAB gpuArray for return. */
	plhs[0] = mxGPUCreateMxArrayOnGPU(out);
	/*
	 * The mxGPUArray pointers are host-side structures that refer to device
	 * data. These must be destroyed before leaving the MEX function.
	 */
	mxGPUDestroyGPUArray(in);
	mxGPUDestroyGPUArray(out);
	mxGPUDestroyGPUArray(rois);
	mxGPUDestroyGPUArray(derin);
    }
    else
    {
	mexErrMsgIdAndTxt(errId, errMsg);
    }
}

