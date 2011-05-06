#include <algorithm>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "node.h"
#include "path.h"

/* includes, project */
#include <cutil_inline.h>

/* includes, kernels */
#include <cppIntegration_kernel.cu>

#define _DEBUG_CUDA (0)

#define cpyMemory(x, y, Sz) do{    \
		memcpy(((void *)(x)),          \
				((const void *)(y)),       \
				((size_t)(Sz)));}while(0)


static volatile bool rdy2malloc = true;

/* Feature HOST-GPU Functions */
extern void gpuAllocFeatures (cudaFeatures &Features) {
#ifdef _DEBUG_FLOAT_GPU
	/* check if input Alpha vector has any bad float values */
	size_t alphatmpsz = Features.cpu.halphaSz/sizeof(float);
	for (size_t i=0;  i < alphatmpsz; i++){
		if (Features.cpu.halpha[i] != Features.cpu.halpha[i]) {
			std::cout << "gpu_gradient, bad alpha value at: " << i << std::endl;
			Features.cpu.halpha[i] = 0.0;
		}
	}
#endif
	cutilSafeCall(cudaMalloc(&(Features.gpu.dAlpha),   Features.cpu.halphaSz));
	cutilSafeCall(cudaMalloc(&(Features.gpu.dfeatures),Features.cpu.hcacheSz));
	cutilSafeCall(cudaMalloc(&(Features.gpu.dbaseFeat),Features.cpu.hfeatureSz));
	cutilSafeCall(cudaMalloc(&(Features.gpu.dexpected),Features.cpu.halphaSz));

}

extern void gpuFreeFeatures (cudaFeatures & Features) {
	delete [] Features.cpu.halpha;
	delete [] Features.cpu.hexpected;
	free(Features.cpu.hfeatures);

	cudaFree(Features.gpu.dAlpha);
	cudaFree(Features.gpu.dexpected);
	cudaFree(Features.gpu.dfeatures);
	cudaFree(Features.gpu.dbaseFeat);
}

extern void gpuCpyFeatures2GPU (cudaFeatures & Features) {
	cutilSafeCall(cudaMemcpy(Features.gpu.dAlpha,Features.cpu.halpha, Features.cpu.halphaSz,
						cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(Features.gpu.dfeatures,Features.cpu.hfeatures, Features.cpu.hcacheSz,
						cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(Features.gpu.dbaseFeat,Features.cpu.hbaseFeat, Features.cpu.hfeatureSz,
						cudaMemcpyHostToDevice));

}

extern void gpuCpyAlpha2GPU (cudaFeatures & Features) {
	cutilSafeCall(cudaMemcpy(Features.gpu.dAlpha,Features.cpu.halpha, Features.cpu.halphaSz,
							cudaMemcpyHostToDevice));
}

extern void gpuClearExpected (cudaFeatures & Features) {
	cudaMemset(Features.gpu.dexpected, 0.0, Features.cpu.halphaSz);
}

extern void gpuCpyExpected2CPU (cudaFeatures * Features) {
	cutilSafeCall(cudaMemcpy(Features->cpu.hexpected,
			Features->gpu.dexpected,
			Features->cpu.halphaSz,
			cudaMemcpyDeviceToHost));
}

extern void gpuCpyExpected2GPU(cudaFeatures * Features) {
	cutilSafeCall(cudaMemcpy(Features->gpu.dexpected,
							 Features->cpu.hexpected,
							 Features->cpu.halphaSz,
							cudaMemcpyHostToDevice));
}

/* Tagger HOST-GPU functions */
extern void gpuAllocTagger (cudaTagger & Tagger) {
		size_t memSz = (sizeof(float) * (Tagger.id.hx) * (Tagger.id.hy));
		size_t memPathSz = (sizeof(float) * (Tagger.id.hx - 1) * (Tagger.id.hy) * (Tagger.id.hy));

		/* path cost: The indexing of the paths in this structure is as
		 * follows:
		 *    idx1D = i*(y^2)+j*(y)+k, where
		 *    i = feature index
		 *    j = output index
		 *    k = path to next output being k
		 *    x is the number of features
		 *    y is the number of possible outputs
		 */
		while(!rdy2malloc);


		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dPathCost),memPathSz));
		cudaMemset(Tagger.gpu.dPathCost, 0, memPathSz);



		/* The following variables occur once per node. The mapping to
		 * this structure is as follows:
		 * 		idx1D = i*(x) + j, where
		 * 		i = feature index
		 * 		j = output index
		 */
		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dNodeCost),memSz));
		cudaMemset(Tagger.gpu.dNodeCost, 0.0, memSz);

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dBestCost),memSz));
		cudaMemset(Tagger.gpu.dBestCost, 0.0, memSz);

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dtrcbck),memSz));
		cudaMemset(Tagger.gpu.dtrcbck, 0.0, memSz);

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dAlpha),memSz));
		cudaMemset(Tagger.gpu.dAlpha, 0.0, memSz);

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dBeta),memSz));
		cudaMemset(Tagger.gpu.dBeta, 0.0, memSz);

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dZ),sizeof(float)));
		cudaMemset(Tagger.gpu.dZ, 0.0, sizeof(float));

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.derr),sizeof(unsigned int)));
		cudaMemset(Tagger.gpu.derr, 0, sizeof(unsigned int));

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.danswers), Tagger.cpu.answersSz));
		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dresults), Tagger.cpu.answersSz));
		cudaMemset(Tagger.gpu.dresults, 0, Tagger.cpu.answersSz);

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.ds),sizeof(float)));
		cudaMemset(Tagger.gpu.ds, 0.0, sizeof(float));

		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dcost),sizeof(float)));
		cudaMemset(Tagger.gpu.dcost, 0.0, sizeof(float));


		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dCpath),memPathSz));
		cutilSafeCall(cudaMalloc(&(Tagger.gpu.dCnode),memSz));

		cutilCheckMsg("Alloc Tagger failed");

		/* Allocate host memory to perform memory copy in contiguous space */
#ifdef COPY_ALL_FROM_GPU
		Tagger.cpu.hPathCost = (float *) malloc(memPathSz);
		assert(Tagger.cpu.hPathCost != NULL);
		Tagger.cpu.hNodeCost = (float *) malloc(memSz);
		assert(Tagger.cpu.hNodeCost != NULL);
		Tagger.cpu.hAlpha    = (float *) malloc(memSz);
		assert(Tagger.cpu.hAlpha != NULL);
		Tagger.cpu.hBeta     = (float *) malloc(memSz);
		assert(Tagger.cpu.hBeta  != NULL);
#endif
		rdy2malloc = false;
}



extern void gpuFreeTagger (cudaTagger & Tagger) {

	while(rdy2malloc);


#ifdef COPY_ALL_FROM_GPU
	free(Tagger.cpu.hPathCost);
	free(Tagger.cpu.hNodeCost);
	free(Tagger.cpu.hAlpha);
	free(Tagger.cpu.hBeta);
#endif
	cudaFree(Tagger.gpu.danswers);
	cudaFree(Tagger.gpu.dresults);
	cudaFree(Tagger.gpu.derr);
	cudaFree(Tagger.gpu.dtrcbck);
	cudaFree(Tagger.gpu.dBestCost);
	cudaFree(Tagger.gpu.dNodeCost);
	cudaFree(Tagger.gpu.dPathCost);
	cudaFree(Tagger.gpu.dAlpha);
	cudaFree(Tagger.gpu.dBeta);
	cudaFree(Tagger.gpu.dZ);
	cudaFree(Tagger.gpu.ds);
	cudaFree(Tagger.gpu.dcost);

	cudaFree(Tagger.gpu.dCnode);
	cudaFree(Tagger.gpu.dCpath);

	cudaThreadSynchronize();

	rdy2malloc = true;
}

/* copy results from GPU to Host */
void gpuCpyTagger2CPU(cudaTagger * Tagger) {
#ifdef COPY_ALL_FROM_GPU
	size_t memSz = (sizeof(float) * (Tagger->id.hx) * (Tagger->id.hy));
	size_t memPathSz = (sizeof(float) * (Tagger->id.hx - 1) * (Tagger->id.hy) * (Tagger->id.hy));


	cutilSafeCall(cudaMemcpy(Tagger->cpu.hPathCost,Tagger->gpu.dPathCost, memPathSz,cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(Tagger->cpu.hNodeCost,Tagger->gpu.dNodeCost, memSz,cudaMemcpyDeviceToHost));


	cutilSafeCall(cudaMemcpy(Tagger->cpu.hAlpha,Tagger->gpu.dAlpha, memSz,cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(Tagger->cpu.hBeta,Tagger->gpu.dBeta, memSz,cudaMemcpyDeviceToHost));
#endif

	cutilSafeCall(cudaMemcpy(&Tagger->cpu.hZ,Tagger->gpu.dZ, sizeof(float),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&Tagger->cpu.hs,Tagger->gpu.ds, sizeof(float),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&Tagger->cpu.herr,Tagger->gpu.derr, sizeof(unsigned int),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&Tagger->cpu.hcost,Tagger->gpu.dcost, sizeof(float),cudaMemcpyDeviceToHost));
}

void gpuCpyAnswers2GPU(cudaTagger *Tagger){
	cutilSafeCall(cudaMemcpy(Tagger->gpu.danswers,Tagger->cpu.answers,Tagger->cpu.answersSz,
			cudaMemcpyHostToDevice));
}


#if _DEBUG_CUDA
static void debug_gradient (cudaFeatures & Features, cudaTagger & Tagger) {
	if ((Tagger.id.hfeature_id_ + 2*Tagger.id.hx - 2) >= Features.cpu.hcacheSz/sizeof(int)){
		std::cout << "ERROR - check feature cache!!!!" << std::endl;
		std::cout << "(CacheSz,featureSz, alphaSz): " <<
				Features.cpu.hcacheSz << "," << Features.cpu.hfeatureSz << "," <<
				Features.cpu.halphaSz << std::endl;
		std::cout << "hfeature_id_, x: " << Tagger.id.hfeature_id_ << " , " <<
				Tagger.id.hx << std::endl;
	}

	if (Tagger.id.hfeature_id_ == 3662){
		for (size_t i = 0; i<Tagger.id.hx; i++){
			std::cout <<"node - fid,xidx: " <<
					(Tagger.id.hfeature_id_ + i) << " , " <<
					i << " -- ";
			std::cout << "fo: " <<
					Features.cpu.hfeatures[Tagger.id.hfeature_id_ + i] << " -- ";
			std::cout << "base: " <<
					Features.cpu.hbaseFeat[Features.cpu.hfeatures[Tagger.id.hfeature_id_ + i]] <<
					std::endl;
		}

		std::cout << std::endl;

		for (size_t i = 1; i<Tagger.id.hx; i++) {
			int pathIdx = (Tagger.id.hfeature_id_ + Tagger.id.hx + (i-1));
			int fo = Features.cpu.hfeatures[pathIdx];
			int base =Features.cpu.hbaseFeat[fo];

			std::cout <<"path - fid,xidx: " << pathIdx << " , " <<
					i << " -- ";
			std::cout << "fo: "   << fo   << " -- ";
			std::cout << "base: " << base << " -- ";
			while (base != -1){
				base = Features.cpu.hbaseFeat[++fo];
				std::cout << base << " , ";
			}
			std::cout << "last yth alpha: " <<
					Features.cpu.halpha[base  + (Tagger.id.hy - 1)*Tagger.id.hy + (Tagger.id.hy - 1)  ] <<
					std::endl;

		}
	}

	std::cout << std::endl;
	std::cout << "gpu_gradient called: ------------------------" << std::endl;
	std::cout << "(CacheSz,featureSz, alphaSz): " <<
			Features.cpu.hcacheSz << "," << Features.cpu.hfeatureSz << "," <<
			Features.cpu.halphaSz << std::endl;
	std::cout << "PATH feature Idx: " << Features.cpu.hfeatures[Tagger.id.hfeature_id_ + 2*Tagger.id.hx - 2] <<
			std::endl;
	int * idx = &Features.cpu.hbaseFeat[Features.cpu.hfeatures[Tagger.id.hfeature_id_ + 2*Tagger.id.hx - 2]];
	std::cout << "PATH *f(0): " << *idx << std::endl;
	std::cout << "Alpha for this idx: " << Features.cpu.halpha[*idx] << std::endl;
	std::cout << "Last Alpha for this idx: " <<
			Features.cpu.halpha[*idx + (Tagger.id.hy-1)*Tagger.id.hy + Tagger.id.hy] << std::endl;

	/* calculate C in debug mode */
	float c = 0.0;
	for (size_t i = 0; i < Tagger.id.hy ; i++) {
		idx = &Features.cpu.hbaseFeat[Features.cpu.hfeatures[Tagger.id.hfeature_id_ + 2*Tagger.id.hx - 2]];
		while (*idx != -1) {
			c += Features.cpu.halpha[*idx + i*Tagger.id.hy + (Tagger.id.hy-1)]; /* calculate cost for y-1 */
			idx++;
		}
		std::cout << "cost for xmax, ymax, path: " << i << " c: " <<
				c << std::endl;
	}

	std::cout << "(x,y,fid) " << Tagger.id.hx<<","<<Tagger.id.hy<<","<<Tagger.id.hfeature_id_ << std::endl;
}
#endif

extern void gpu_gradient(cudaFeatures & Features, cudaTagger & Tagger) {



	/* CUDA memory allocation and initialization. This is equivalent
	 * to the buildlattice function.
	 */
#if _DEBUG_CUDA
	debug_gradient(Features,Tagger);
	cudaPrintfInit();
#endif

	/* setup execution parameters and build lattice*/
	dim3 grid(0, 0, 0);
	dim3 threads(0, 0, 0);
	calcBLKThreads((Tagger.id.hx*Tagger.id.hy),grid,threads);
	buildlatticekernel <<< grid.x, threads.x >>> (Tagger.gpu,Tagger.id,Features.gpu);
	/* check if kernel execution generated an error */
	cutilCheckMsg("Kernel buildlatticekernel failed");
	cutilSafeThreadSync();

#ifdef GPU_SEQ_PATH_COST
	pathCost <<< 1, 1 >>> (Features.gpu.dAlpha,
				(Features.cpu.halphaSz/sizeof(float)),
				Features.gpu.dfeatures,
				Features.gpu.dbaseFeat,
				Tagger.gpu.dPathCost,
				Tagger.gpu.dNodeCost,
				Tagger.id.hx,
				Tagger.id.hy,
				Tagger.id.hfeature_id_);
	cutilCheckMsg("pathCost failed");
#endif

	/* Perform Alpha Beta Sum */
	grid.x = grid.y = grid.z = 1;
	threads.x = Tagger.id.hy;
	threads.y = threads.z = 1;
	forwardBackwardKernel <<< grid.x, threads.x >>> (Tagger.gpu,Tagger.id,Features.gpu);
	/* check if kernel execution generated an error */
	cutilCheckMsg("Kernel forwardBackwardKernel failed");
	cutilSafeThreadSync();

#ifdef PARALELL_EXP
	calcBLKThreads((Tagger.id.hx*Tagger.id.hy),grid,threads);
	calcCKernel <<< grid.x, threads.x >>> (Tagger.gpu,Tagger.id,Features.gpu);
	/* check if kernel execution generated an error */
	cutilCheckMsg("Kernel calcCKernel failed");
	cutilSafeThreadSync();
#endif


#ifdef GPU_PARALELL_EXPECTATION
	calcBLKThreads((Tagger.id.hx*Tagger.id.hy),grid,threads);
	cudaPrintfInit();
	calcExpectationKernel <<< grid.x, threads.x >>> (Tagger.gpu,Tagger.id,Features.gpu);
	cutilCheckMsg("Kernel calcExpectationKernel failed");
	cutilSafeThreadSync();
#else /* Expectaction executed in a sequential manner to avoid
 	 	 write on write problems with the expectation vector.
 	 	*/

	calcExpectationKernelSingle <<< 1, 1 >>> (Tagger.gpu,Tagger.id,Features.gpu);
	/* check if kernel execution generated an error */
	cutilCheckMsg("Kernel calcExpectationKernelSingle failed");
	cutilSafeThreadSync();
#endif

	calcSSingle <<< 1, 1 >>> (Tagger.gpu,Tagger.id,Features.gpu);
	/* check if kernel execution generated an error */
	cutilCheckMsg("Kernel calcSSingle failed");
	cutilSafeThreadSync();


	cu_viterbi <<< 1, Tagger.id.hy >>> (Tagger.gpu,Tagger.id,Features.gpu);
	/* check if kernel execution generated an error */
	cutilCheckMsg("Kernel cu_viterbi failed");
	cutilSafeThreadSync();


#if _DEBUG_CUDA
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
#endif

}

