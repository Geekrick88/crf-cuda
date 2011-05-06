/*
 * crf_cuda_integration.h
 *
 *  Created on: Mar 24, 2011
 *      Author: Miguel Osorio
 */

#ifndef CRF_CUDA_INTEGRATION_H_
#define CRF_CUDA_INTEGRATION_H_
#include <vector>
#include "node.h"
#include "path.h"


typedef struct {
	/*!> Alpha vector */
	float * halpha;
	/*!> Alpha vector size */
	size_t halphaSz;
	/*!> feature indexes */
	int * hfeatures;
	/*!> feature indexes size */
	size_t hcacheSz;
	/*!> base features indexing alpha vector */
	int * hbaseFeat;
	/*!> base features size */
	size_t hfeatureSz;
	/*!> Expectation Vector */
	float * hexpected;
}CPUcudaFeatures;


typedef struct {
	/*! GPU pointers*/
	/*!> GPU Alpha vector */
	float * dAlpha;
	/*!> Feature indexes */
	int * dfeatures;
	/*!> base features indexing to alpha vector */
	int * dbaseFeat;
	/*!> Expectation Vector */
	float * dexpected;
}GPUcudaFeatures;

typedef struct  {
	CPUcudaFeatures cpu;
	GPUcudaFeatures gpu;
}cudaFeatures;





typedef struct {
	/*! Host pointers to linearize
	 *  Tagger data.  */
	/*!> node's path cost*/
	float *hPathCost;
	/*!> node's cost */
	float *hNodeCost;
	/*!> Node's Alpha */
	float *hAlpha;
	/*!> Node's Beta */
	float *hBeta;
	/*!> Cumulative expectation*/
	float hZ;

	float hs;
	float hcost;
	unsigned int herr;

	/*!> Answers vector */
	unsigned short * answers;
	/*!> Size of answers vector*/
	size_t answersSz;
}CPUcudaTagger;


typedef struct {
	/*! GPU pointers */
	unsigned short * danswers;
	/*! results produced by tagger */
	unsigned short * dresults;
	/*! tagger error */
	unsigned int *derr;
	/*! traceback pointer */
	unsigned int *dtrcbck;
	/*! best cost used by viterbi */
	float *dBestCost;
	/*! node's cost */
	float *dNodeCost;
	/*! node's alpha */
	float *dAlpha;
	/*! node's beta */
	float *dBeta;
	/*! path cost */
	float *dPathCost;
	/*! expectation */
	float *dZ;
	/*! expectation - answer */
	float *ds;
	/*! viterbi cost */
	float *dcost;

	/*! c-node - used in expectation calc*/
	float *dCnode;
	/*! c-path - used in expectation calc*/
	float *dCpath;
}GPUcudaTagger;


typedef struct {
	/*!> Tagger ID */
	size_t hfeature_id_;
	/*!> number of rows in node map, corresponds
	 * to units within tagger structure. */
	size_t hx;
	/*!> number of columns in node map,
	 * corresponds to possible outputs.
	 */
	size_t hy;
}cuTaggerID;

typedef struct {
	cuTaggerID    id;
	CPUcudaTagger cpu;
	GPUcudaTagger gpu;
} cudaTagger ;

#endif /* CRF_CUDA_INTEGRATION_H_ */
