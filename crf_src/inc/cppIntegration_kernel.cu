/**
 * Add License here
 *
 * \author Miguel A Osorio:  mao2130 {at} columbia {dot} edu
 *
 * \history
 *
 * 3/14/2011 M.Osorio Implemented buildlatticekernel using global memory
 */

#ifndef _CPP_INTEGRATION_KERNEL_H_
#define _CPP_INTEGRATION_KERNEL_H_


#include "cuPrintf.cu"
#include "crf_cuda_integration.h"

//#define PARALELL_EXP (1)
#define GPU_PARALELL_EXPECTATION (1)

/* i = part of speech, j = youtput associated with i, k = youtput for i+1 */
#define GO_RIGHT(ySq,y,i,j,k) (((i)*(ySq))+((j)*(y))+(k)) /*macro to get next right path*/
#define GO_LEFT(ySq,y,i,j,k)  ((((i)-1)*(ySq))+((k)*(y))+(j)) /*macro to get next right path*/

#define MAX_THREADS_PER_BLOCK (480)
#define MAX_BLOCKS			  (8) /* This parameter can vary */

#define GPU_cost_factor_ (float)(1.0)
#define cuMINUS_LOG_EPSILON  50   /* used in logsumexp function */

void calcBLKThreads(unsigned int numNodes, dim3 &blockGrid, dim3 &threadGrid){

	size_t numBlocks = numNodes / MAX_THREADS_PER_BLOCK;
	size_t remainder = numNodes % MAX_THREADS_PER_BLOCK;
	size_t numThreads = MAX_THREADS_PER_BLOCK;

	if (remainder > 0){
		numBlocks += 1;
		/* Change number of threads per block here to balance the load */
		numThreads = numNodes/numBlocks;
		if ((numNodes%numBlocks)>0)
			numThreads += 1;
	}

	/* return flat dim for the kernel - Remember that the
	 * block and thread dimensions are converted into arrays by GPU
	 * scheduler, so it does not matter which format we use.
	 */
	blockGrid.x = numBlocks;
	blockGrid.y = blockGrid.z = 0;
	threadGrid.x = numThreads;
	threadGrid.y = threadGrid.z = 0;
}

/**
 * This function calculates the cost for all the input nodes. It follows
 * the same data patters used in the CRF++ tagger & feature index implementation.
 *
 * This kernel makes the hard assumption that the number of threads combined with
 * the number of blocks, will be equivalent to the number of nodes for which we need
 * to calculate the costs.
 *
 *\param ft.dAlpha Pointer to Alpha feature set
 *\param ft.dfeatures Feature indexes to feature alpha array.
 *\param ft.dbaseFeat dFeature memory offset for each node
 *\param tgr.dPathCost
 *\param tgr.dNodeCost
 *\param x
 *\param y
 *
 *
 *\todo Need to make sure that tagger fid is always constant.
 *\todo Need a way to pass fid;
 *\todo redefine ft.dAlpha and ft.dfeatures as constants, given that this is
 *		calculated outside the GPU
 *
 *
 */
__global__ void buildlatticekernel(GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{

	int idxFeat = blockDim.x * blockIdx.x + threadIdx.x;

	/* TODO: need to convert the id.hx * id.hy computation into a constant value to
	 * speed things up. */
	if (idxFeat >= ((id.hy) * (id.hx)))
		return;



	/* need to calculate the index to y -i.e output for current node */
	int y = idxFeat % (id.hy);

	/* get feature offset */
	/* here is the problem, idxFeat has to be equivalent to the node.x number
	 * idxFeat is just the global number.*/
	int x = idxFeat / (id.hy);


	if ((x >= id.hx) || (y >= id.hy))
		return;

	int featOffset = ft.dfeatures[id.hfeature_id_ + x];

	/* ft.dbaseFeat and ft.dfeatures */
	float c = 0.0;

	int f = ft.dbaseFeat[featOffset];
	while(f != -1){
		c += ft.dAlpha[f + y];
		f = ft.dbaseFeat[++featOffset];
	}

	/* ignore cost factor for now - There is an actual cost factor used in
	 * the CRF++ implementation.
	 */
	tgr.dNodeCost[idxFeat] = c;

	/* Per Path calculation. The feature index only changes as a
	 * function of X, and the initial value is equal to X.  In the
	 * CPU implementation, Alpha is obtained by the following sum:
	 * 	(A)[*f + p->lnode->y * y_.size() + p->rnode->y];*/

	/* how way we get f needs to change. need to get rpath f for x-1 */
	if (x > 0) {
		int xLdysq = (x-1)*id.hy*id.hy;
		/* id.hfeature_id_ is the base for this tagger, which points
		 * to node i = 0, j = 0. The first path alpha is located
		 * at id.hfeature_id_ + (id.hx), and we want the lnode alpha
		 * which is the right node of (x-1). The equivalent
		 * expression is executed below.*/
		featOffset = ft.dfeatures[id.hfeature_id_ + id.hx + (x - 1)];
		int tmpfOffset = 0;
		for (int i = 0; i < id.hy; i++) {
			c = 0.0;
			tmpfOffset = featOffset;
			f = ft.dbaseFeat[tmpfOffset];

			while (f >= 0) {
				c += ft.dAlpha[f + (i*id.hy) + y];
				f  = ft.dbaseFeat[++tmpfOffset];
			}
			tgr.dPathCost[xLdysq + (i*id.hy) + y] = c;
		}
	}
}

__global__ void pathCost (GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{
	int printEn = 0;

	if (id.hfeature_id_==3662)
		printEn = 1;

	for (unsigned int cur = 1; cur < id.hx; cur++) {
		for (unsigned int j = 0; j < id.hy; j++) {
			int xLdysq = (cur-1)*id.hy*id.hy;
			if(printEn)cuPrintf("\n\ny,x,fid: %d,%d,%d\n", id.hy,id.hx,id.hfeature_id_);
			int cacheidx = id.hfeature_id_ + id.hx + (cur - 1);
			if(printEn)cuPrintf("\ncacheidx: %d ",cacheidx);
			int featOffset = ft.dfeatures[cacheidx];
			if(printEn)cuPrintf("\nfo: %d \nf: ",featOffset);
			int tmpfOffset = 0;
			for (int i = 0; i < id.hy; i++) {
				float c = 0.0;
				tmpfOffset = featOffset;
				int f = ft.dbaseFeat[tmpfOffset];
				while (f >= 0) {
					if(printEn)cuPrintf(":%d: ",f);
					c += ft.dAlpha[f + (i*id.hy) + j];
					f  = ft.dbaseFeat[++tmpfOffset];
				}
				tgr.dPathCost[xLdysq + (i*id.hy) + j] = c;
			}
		}
	}
}


__device__ float _cuLogsumExp (float x, float y, bool flg)
{
	if (flg) return y;

	float min = y, max = x;
	if (x < y) {
		min = x;
		max = y;
	}
	if (max > min + cuMINUS_LOG_EPSILON) {
		return max;
	} else {
		return (max) + logf(expf(min - max) + 1.0);
	}
}

/**
 * Alpha and Beta Calculation considerations:
 * The CPU code calculates Alpha and Beta for each node.
 *
 *<h2>Algorithm:</h2>
 *
 * For each node:
 *  For each path in lpath:
 *  	alpha += sumexp( cost + lpath.cost ).
 *  alpha += cost.
 *
 * Y rows, per X columns.
 * - Each node will have Y left path nodes, except for the first column.
 * - Each column can be calculated in parallel, but will block the execution
 *   of the next column iteration.
 *
 * So, it sounds that a Y x Y kernel will be able to do this job, but this will
 * require synchronization between iterations.
 *
 * Note: Only run this kernel inside 1 block.
 *
 * Alpha and Beta could be calculated in Parallel.
 */

/* Parallel version of forward-backward algorithm */
__global__ void forwardBackwardKernel(GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{

	if (threadIdx.x >= id.hy)
		return;

	/* initialize alpha for first node column */
	tgr.dAlpha[threadIdx.x] = tgr.dNodeCost[threadIdx.x];

	/* initialize beta for last node column */
	tgr.dBeta[(id.hx-1)*id.hy + threadIdx.x] = tgr.dNodeCost[(id.hx-1)*id.hy + threadIdx.x];

	__syncthreads();

	unsigned int irev = id.hx - 2;
	/* Important to note that there is one thread per each Y
	 * There are (x-1) iterations.*/
	for (unsigned int i = 1; i < id.hx; i++) {
		float alpha = 0.0; float beta = 0.0;
		float nodeCostA = tgr.dNodeCost[i*id.hy + threadIdx.x];
		float nodeCostB = tgr.dNodeCost[irev*id.hy + threadIdx.x];
		for (unsigned int j = 0; j < id.hy; j++) {
			/* get path cost for alpha */
			float pathCostA = tgr.dPathCost[GO_LEFT((id.hy*id.hy),id.hy,i,threadIdx.x,j)];
			/* get path cost for beta */
			float pathCostB = tgr.dPathCost[GO_RIGHT((id.hy*id.hy),id.hy,irev,threadIdx.x,j)];
			alpha = _cuLogsumExp(alpha, pathCostA +
					tgr.dAlpha[(i-1)*id.hy + j], (j == 0));
			beta = _cuLogsumExp(beta, pathCostB +
					tgr.dBeta[(irev+1)*id.hy + j],(j == 0));
		}
		tgr.dAlpha[i*id.hy + threadIdx.x] = alpha + nodeCostA;
		tgr.dBeta[irev*id.hy + threadIdx.x] = beta + nodeCostB;
		irev--;
		__syncthreads();
	}



	/* TODO: optimize this later*/
	if (threadIdx.x == 0) {
		float Z = 0.0;
		for (unsigned int i = 0; i < id.hy; i++ )
			Z = _cuLogsumExp(Z,tgr.dBeta[i],i==0);
		*tgr.dZ = Z;
	}
}

__device__ inline void atomicFloatAdd(float *address, float val)
{
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;

	while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
	{
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}
}

/** Calculate expectation for each node.
 *
 *  \note use one thread per node. */
__global__ void calcExpectationKernel (GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{
	/* Get Idx for this block */
	int idxFeat = blockDim.x * blockIdx.x + threadIdx.x;

	/* Only allow threads that fall inside x * y matrix */
	if (idxFeat >= ((id.hy) * (id.hx)))
		return;

	/* need to calculate the index to y -i.e output for current node */
	int y = idxFeat % (id.hy);

	/* get feature offset */
	/* here is the problem, idxFeat has to be equivalent to the node.x number
	 * idxFeat is just the global number.*/
	int x = idxFeat / (id.hy);
	int xy = x * id.hy;

	if ((x >= id.hx) || (y >= id.hy))
		return;

	int featOffset = ft.dfeatures[id.hfeature_id_ + x];

	/* ft.dbaseFeat and ft.dfeatures */
	float c = expf((tgr.dAlpha[xy + y]) + (tgr.dBeta[xy + y]) -
			(tgr.dNodeCost[xy + y]) - (*tgr.dZ));

	/* this implements node expectation calculation */
	int f = ft.dbaseFeat[featOffset];
	while(f != -1){
		atomicFloatAdd(&ft.dexpected[f + y], c);
		f = ft.dbaseFeat[++featOffset];
	}

	/* the code below calculates the expectation values for all paths */
	/* how way we get f needs to change. need to get rpath f for x-1 */
	if (x > 0) {
		int xLdysq = (x-1)*id.hy*id.hy;
		int xLdy   = (x-1)*id.hy;
		/* id.hfeature_id_ is the base for this tagger, which points
		 * to node i = x, j = y. The first path alpha is located
		 * at id.hfeature_id_ + (id.hx), and we want the lnode alpha
		 * which is the right node of (x-1). The equivalent
		 * expression is executed below.*/
		featOffset = ft.dfeatures[id.hfeature_id_ + id.hx + x - 1];
		for (int i = 0; i < id.hy; i++) {
			int tmpfOffset = featOffset;
			f = ft.dbaseFeat[tmpfOffset];
			/* exp(lnode->alpha + cost + rnode->beta - Z)*/
			c = expf((tgr.dAlpha[xLdy+i]) + (tgr.dPathCost[xLdysq + (i*id.hy) + y]) +
					(tgr.dBeta[xy + y]) - (*tgr.dZ));

			while (f >= 0) {
				atomicFloatAdd(&ft.dexpected[f + (i*id.hy) + y],c);
				f = ft.dbaseFeat[++tmpfOffset];
			}
		}
	}
}


__global__ void calcCKernel(GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{
	/* Get Idx for this block */
	int idxFeat = blockDim.x * blockIdx.x + threadIdx.x;

	/* Only allow threads that fall inside x * y matrix */
	if (idxFeat >= ((id.hy) * (id.hx)))
		return;

	/* need to calculate the index to y -i.e output for current node */
	int y = idxFeat % (id.hy);

	/* get feature offset */
	/* here is the problem, idxFeat has to be equivalent to the node.x number
	 * idxFeat is just the global number.*/
	int x = idxFeat / (id.hy);
	int xy = x * id.hy;

	/* node C */
	float c = expf((tgr.dAlpha[x*id.hy + y]) + (tgr.dBeta[x*id.hy + y]) -
			(tgr.dNodeCost[x*id.hy + y]) - (*tgr.dZ));
	tgr.dCnode[xy+y] = c;

	/* path C */
	/* the code below calculates the expectation values for all paths */
	/* how way we get f needs to change. need to get rpath f for x-1 */
	if (x > 0) {
		int xLdysq = (x-1)*id.hy*id.hy;
		int xLdy   = (x-1)*id.hy;
		/* id.hfeature_id_ is the base for this tagger, which points
		 * to node i = x, j = y. The first path alpha is located
		 * at id.hfeature_id_ + (id.hx), and we want the lnode alpha
		 * which is the right node of (x-1). The equivalent
		 * expression is executed below.*/
		for (int i = 0; i < id.hy; i++) {
			/* exp(lnode->alpha + cost + rnode->beta - Z)*/
			c = expf((tgr.dAlpha[xLdy+i]) + (tgr.dPathCost[xLdysq + (i*id.hy) + y]) +
					(tgr.dBeta[x*id.hy + y]) - (*tgr.dZ));
			tgr.dCpath[xLdysq + (i*id.hy) + y] = c;
		}
	}
}



/* Write on write errors are crippling these measurements, try to execute this using only
 * one kernel
 */
__global__ void calcExpectationKernelSingle(GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{
	for (unsigned int cur = 0; cur < id.hx ; cur++){
		for (unsigned int j = 0; j < id.hy; j++){
			int featOffset = ft.dfeatures[id.hfeature_id_ + cur];

			/* ft.dbaseFeat and ft.dfeatures */
#ifdef PARALELL_EXP
			float c = tgr.dCnode[cur*id.hy + j];
#else
			float c = expf((tgr.dAlpha[cur*id.hy + j]) + (tgr.dBeta[cur*id.hy + j]) -
					(tgr.dNodeCost[cur*id.hy + j]) - (*tgr.dZ));
#endif

			/* this implements node expectation calculation */
			int f = ft.dbaseFeat[featOffset];
			while(f != -1){
				atomicFloatAdd(&ft.dexpected[f + j],c);
				f = ft.dbaseFeat[++featOffset];
			}

			/* the code below calculates the expectation values for all paths */
			/* how way we get f needs to change. need to get rpath f for x-1 */
			if (cur > 0) {
				int xLdysq = (cur-1)*id.hy*id.hy;
#ifndef PARALELL_EXP
				int xLdy   = (cur-1)*id.hy;
#endif
				/* id.hfeature_id_ is the base for this tagger, which points
				 * to node i = x, j = y. The first path alpha is located
				 * at id.hfeature_id_ + (id.hx), and we want the lnode alpha
				 * which is the right node of (x-1). The equivalent
				 * expression is executed below.*/
				featOffset = ft.dfeatures[id.hfeature_id_ + id.hx + cur - 1];
				for (int i = 0; i < id.hy; i++) {
					int tmpfOffset = featOffset;
					f = ft.dbaseFeat[tmpfOffset];
					/* exp(lnode->alpha + cost + rnode->beta - Z)*/
#ifdef PARALELL_EXP
					c = tgr.dCpath[xLdysq + (i*id.hy) + j];
#else
					c = expf((tgr.dAlpha[xLdy+i]) + (tgr.dPathCost[xLdysq + (i*id.hy) + j]) +
							(tgr.dBeta[cur*id.hy + j]) - (*tgr.dZ));
#endif

					while (f >= 0) {
						ft.dexpected[f + (i*id.hy) + j] += c;
						f = ft.dbaseFeat[++tmpfOffset];
					}
				}
			}
		}
	}
}




__global__ void calcSSingle(GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{
	/* s parameter calculation */
	*tgr.ds = 0.0;

	for (unsigned int i = 0; i < id.hx; i++) {
		if (tgr.danswers[i]>=id.hy)
			cuPrintf("answer out of bounds\n\n");
		int featOffset = ft.dfeatures[id.hfeature_id_ + i];
		int f = ft.dbaseFeat[featOffset];
		while(f >= 0){
			float tmp = ft.dexpected[f + (unsigned int)tgr.danswers[i]] - 1.0;
			ft.dexpected[f + (unsigned int)tgr.danswers[i]] = tmp;

			f = ft.dbaseFeat[++featOffset];
		}
		*tgr.ds += (tgr.dNodeCost[i*id.hy + (unsigned int)tgr.danswers[i]]); /* UNIGRAM cost */
		if (i>0){
			unsigned int j = (unsigned int)tgr.danswers[(i-1)];
			featOffset = ft.dfeatures[id.hfeature_id_ + id.hx + (i-1)];
			f = ft.dbaseFeat[featOffset];
			while(f >= 0){
				float tmp = ft.dexpected[f + (j*id.hy) + (unsigned int)tgr.danswers[i]] - 1.0;
				ft.dexpected[f + (j*id.hy) + (unsigned int)tgr.danswers[i]] = tmp;
				f = ft.dbaseFeat[++featOffset];
			}
			*tgr.ds += (tgr.dPathCost[(i-1)*id.hy*id.hy + j*id.hy + (unsigned int)tgr.danswers[i]]);

		}
	}
}




/* to be executed by Yn threads in parallel*/
__global__ void cu_viterbi(GPUcudaTagger tgr, cuTaggerID id, GPUcudaFeatures ft)
{
	if (threadIdx.x >= id.hy)
		return;

	unsigned int dy2sq = ((id.hy) * (id.hy));

	/* initialize alpha for first node column */
	tgr.dBestCost[threadIdx.x] = tgr.dNodeCost[threadIdx.x];



	float        costmp; /* local memory */
	int maxIdx = 0;

	/* Important to note that there is one thread per each Y
	 * There are (x-1) iterations.*/
	for (unsigned int i = 1; i < id.hx; i++) {
		__syncthreads();
		float bestc = -1e37;
		maxIdx = -1;
		unsigned int xysq = (i-1)*dy2sq;
		unsigned int xy   = (i)*id.hy;
		for (unsigned int j = 0; j < id.hy; j++) {
			costmp = tgr.dBestCost[(i-1)*id.hy + j] +
				  tgr.dPathCost[xysq + j*id.hy + threadIdx.x] +
				  tgr.dNodeCost[xy + threadIdx.x];
			if(costmp > bestc){
				bestc  = costmp;
				maxIdx = j;
			}
		}
		tgr.dtrcbck[xy + threadIdx.x]   = maxIdx;
		if(maxIdx == -1){
			tgr.dBestCost[xy + threadIdx.x] = \
					tgr.dNodeCost[xy + threadIdx.x];
		} else {
			tgr.dBestCost[xy + threadIdx.x] = bestc;
		}
	}


	__syncthreads();
	/* find best path */
	if(threadIdx.x == 0) {
		float bestc = -1e37;
		maxIdx = 0;
		unsigned int s = (id.hx-1)*id.hy;
		for (size_t j = 0; j < id.hy; j++) {
			if (bestc < tgr.dBestCost[s+j]) {
				maxIdx = j;
				bestc = tgr.dBestCost[s+j];
			}
		}
		tgr.dresults[id.hx-1] = maxIdx;
		for (int x = (id.hx-2);x>=0; x--)
			tgr.dresults[x] = tgr.dtrcbck[(x+1)*id.hy+tgr.dresults[(x+1)]];

		*tgr.dcost = -bestc;

		/*calculate errors - optimize this later */
		unsigned int err = 0;
		for (unsigned int x = 0; x < id.hx; x++){
			if(tgr.danswers[x] != tgr.dresults[x])
				err++;
		}
		*tgr.derr = err;
	}


}
#endif /* #ifndef _CPP_INTEGRATION_KERNEL_H_ */
