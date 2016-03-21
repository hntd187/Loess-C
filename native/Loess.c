#include <jni.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "com_scarman_loess_Loess.h"

void updateBandwidth(jdouble* xVals, jdouble* weights, int length, int i, int* bandwidth);
int nextNonZero(jdouble* weights, int i, int length);
double tricube(double x);

JNIEXPORT jdoubleArray JNICALL Java_com_scarman_loess_Loess_smooth(
    JNIEnv* env,
    jobject thisObj,
    jdoubleArray xVals,
    jdoubleArray yVals,
    jdoubleArray weights,
    jdouble bandwidth,
    jint robustnesstIters) {

    double *xArray = (*env)->GetDoubleArrayElements(env, xVals, NULL);
    double *yArray = (*env)->GetDoubleArrayElements(env, yVals, NULL);
    double *wArray = (*env)->GetDoubleArrayElements(env, weights, NULL);
    
    double accuracy = 1E-12;
    
    if(NULL == xArray || NULL == yArray){
        return NULL;
    }
    
    int n = (*env)->GetArrayLength(env, xVals);
    int wlength = (*env)->GetArrayLength(env, weights);
    
    int bandwidthInPoints = (int)(bandwidth * n);
    
    double res[n];
    double residuals[n];
    double sortedResiduals[n];
    double robustnessWeights[n];
    
    int iter, r, i, k;
    for (r = 0; r < n; r++){
        robustnessWeights[r] = 1.0;
    }
    
    for(iter = 0; iter <= robustnesstIters; iter++) {
        int bandwidthInterval[] = {0, bandwidthInPoints - 1};
        
        for (i = 0; i < n; i++){
            double x = xArray[i];
            if (i > 0){
                updateBandwidth(xArray, wArray, wlength, i, bandwidthInterval);
            }
            int ileft = bandwidthInterval[0];
            int iright = bandwidthInterval[1];
            
            int edge;
            if (xArray[i] - xArray[ileft] > xArray[iright] - xArray[i]){
                edge = ileft;
            } else {
                edge = iright;
            }
            
            double sumWeights = 0.0;
            double sumX = 0.0;
            double sumXSquared = 0.0;
            double sumY = 0.0;
            double sumXY = 0.0;
            double denom = abs(1.0 / (xArray[edge] - x));
            
            for(k = 0; k <= iright; k++){
                double xk   = xArray[k];
                double yk   = xArray[k];
                double dist = (k < i) ? x - xk : xk - x;
                double w    = tricube(dist * denom) * robustnessWeights[k] * wArray[k];
                double xkw  = xk * w;
                sumWeights += w;
                sumX += xkw;
                sumXSquared += xk * xkw;
                sumY += yk * w;
                sumXY += yk * xkw;
            }
            
            double meanX = sumX / sumWeights;
            double meanY = sumY / sumWeights;
            double meanXY = sumXY / sumWeights;
            double meanXSquared = sumXSquared / sumWeights;
            
            double beta;
            if (sqrt(abs(meanXSquared - meanX * meanX)) < accuracy) {
                beta = 0.0;
            } else {
                beta = (meanXY - meanX * meanY) / (meanXSquared - meanX * meanX);
            }
            double alpha = meanY - beta * meanX;
            res[i] = beta * x + alpha;
            residuals[i] = abs(yArray[i] - res[i]);
        }
        
        if (iter == robustnesstIters){
            break;
        }
        
        memcpy(residuals, sortedResiduals, sizeof(sortedResiduals));
        qsort(sortedResiduals);
        double medianResidual = sortedResiduals[n / 2];
        
        if (abs(medianResidual) < accuracy){
            break;
        }
        
        for(i = 0; i < n; i++){
            double arg = residuals[i] / (6.0 * medianResidual);
            if (arg >= 1.0){
                robustnessWeights[i] = 0.0;
            } else {
                double w = 1.0 - arg * arg;
                robustnessWeights[i] = w * w;
            }
        }
    }
    

    jdoubleArray outArray = (*env)->NewDoubleArray(env, n);
    if(NULL == outArray){
        return NULL;
    }
    
    (*env)->SetDoubleArrayRegion(env, outArray, 0, n, res);
    
    (*env)->ReleaseDoubleArrayElements(env, xVals, xArray, 0);
    (*env)->ReleaseDoubleArrayElements(env, yVals, yArray, 0);
    (*env)->ReleaseDoubleArrayElements(env, weights, wArray, 0);
    return outArray;
}

void updateBandwidth(double* xVals, double* weights, int length, int i, int* bandwidth){
    int left = bandwidth[0];
    int right = bandwidth[1];
    int nextRight = nextNonZero(weights, right, length);
    if (nextRight < length && xVals[nextRight] - xVals[i] < xVals[i] - xVals[left]){
        int nextLeft = nextNonZero(weights, bandwidth[0], length);
        bandwidth[0] = nextLeft;
        bandwidth[1] = nextRight;
    }
}

int nextNonZero(jdouble* weights, int i, int length){
    int j = i + 1;
    while (j < length && weights[j] == 0.0){
        j++;
    }
    return j;
}

double tricube(double x){
    double absX = abs(x);
    if (absX >= 1.0){
        return 0.0;
    }
    double triX = 1 - absX * absX * absX;
    return (triX * triX * triX);
}