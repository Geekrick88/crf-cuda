/*
 * crf_cuda.h
 *
 *  Created on: Mar 4, 2011
 *      Author: PNTX43
 */

#ifndef CRF_CUDA_H_
#define CRF_CUDA_H_

struct cuPath;

struct cuNode {
	unsigned int x;
	unsigned int y;
	double       alpha; //This will be casted to float by the GPU
	double 		 beta;
	double 		 cost;
	double       bestCost;
	cuNode		 *prev;
	int          *fvector;
	std::vector<cuPath *> lpath;
	std::vector<cuPath *> rpath;
};

struct cuPath {
	cuNode *rnode;
	cuNode *lnode;
	int    *fvector;
	double  cost;
};



/* need to add macros to initialize path and node */

#endif /* CRF_CUDA_H_ */
