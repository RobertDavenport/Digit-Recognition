import java.lang.Math;
import java.util.Arrays;

/* Desc */
class DigitRecognition
{
    // Input Layer
    public static double[][] activationLayer0 = 
    {
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {0, 0, 1, 1},
        {1, 1, 0, 0}
    };

    // Weights
    public static double[][] weightLayer1 = 
    {
        {-.21, .72, -.25, 1},
        {-.94, -.41, -.47, .63},
        {.15, .55, -.49, -.75}
    };

    public static double[][] weightLayer2 =
    {
        {.76, .48, -.73},
        {.34, .89, -.23}
    };

    // Bias
    public static double[] biasLayer1 = {.1, -.36, -.31};
    public static double[] biasLayer2 = {.16, -.46,};

    // Classification
    public static double[][] classifcationSet = 
    {   
        {0, 1},
        {1, 0},
        {0, 1},
        {1, 0}
    };

    public static int miniBatchSize = 2;
    public static int eta = 10;
    public static int miniBatchPerEpoch = (activationLayer0.length / miniBatchSize);

    // 3D arrays for storing calculated weights for each run of a minibatch
    public static double[][][] calculatedWeightsLayer1 = new double[miniBatchSize][weightLayer1.length][weightLayer1[0].length];
    public static double[][][] calculatedWeightsLayer2 = new double[miniBatchSize][weightLayer2.length][weightLayer2[0].length];
    // 3D arrays for storing calculated bias for each run of a minibatch
    public static double[][] calculatedBiasesLayer1 = new double[miniBatchSize][biasLayer1.length];
    public static double[][] calculatedBiasesLayer2 = new double[miniBatchSize][biasLayer2.length];


    public static double[] calculateZLayer(double[] activationLayer, double[][] weightLayer, double[] biasLayer) {
        double[] zLayer = new double[biasLayer.length];       
        for (int i = 0; i < biasLayer.length; i++) {
            double z = 0;
            for (int j = 0; j < activationLayer.length; j++)
            {
                z += weightLayer[i][j] * activationLayer[j];
            }
            zLayer[i] = z + biasLayer[i];
        }
        return zLayer;
    }

    public static double[] calculateALayer(double[] zLayer){
        double[] aLayer = new double[zLayer.length];
        for (int i = 0; i < zLayer.length; i++) {
            aLayer[i] = 1 / (1 + Math.pow(Math.E, -zLayer[i]));
        }
        return aLayer;
    }

    public static double calculateCost(double[] aLayer, double[] classifications)
    {
        return (.5) * ((Math.pow((classifications[0] - aLayer[0]), 2)) + (Math.pow((classifications[1] - aLayer[1]), 2)));
    }

    public static double[] backwardPorpigationLayer1(double[][] weights, double[] gradientBias, double[] aLayer)
    {
        double[] gradientbias = new double[aLayer.length];
        for (int i = 0; i < weights[0].length; i++)
        {
            for (int j = 0; j < weights.length; j++)
            {
                gradientbias[i] += weights[j][i] * gradientBias[j];
            }
            gradientbias[i] *= (aLayer[i] * (1 - aLayer[i]));
        }

        return gradientbias;
    }

    public static double[] backwardPorpigationLayer2(double[] aLayer, double[] classifications)
    {
        double[] gradientbias = new double[aLayer.length];
        for (int i = 0; i < aLayer.length; i++){
            gradientbias[i] = (aLayer[i] - classifications[i]) * aLayer[i] * (1 - aLayer[i]);
        }
        return gradientbias;
    }

    public static double[][] calculateWeightGradient(double[] aLayer, double[] gradientBias)
    {
        double[][] weightGradient = new double[gradientBias.length][aLayer.length];
        for (int i = 0; i < gradientBias.length; i++)
        {
            for (int j = 0; j < aLayer.length; j++)
            {
                weightGradient[i][j] = aLayer[j] * gradientBias[i];
            }
        }
        return weightGradient;
    }

    public static double[] reviseBias(double[][] biasGradient, double[] startingBias)
    {
        double[] revisedBias = new double[startingBias.length];       

        for (int i = 0; i < startingBias.length; i++)
        {
            double sumBiasGradient = 0;
            for (int j = 0; j < biasGradient.length; j++)
            {
                sumBiasGradient += biasGradient[j][i]; 
            }
            revisedBias[i] = startingBias[i] - (eta/2) * sumBiasGradient;         
        }
        return revisedBias;
    }

    // Similar to reviseBias but altered to handle the 3D array used to store the weight gradients
    public static double[][] reviseWeights(double[][][] weightGradient, double[][] startingWeights)
    {
        double[][] revisedWeights = new double[startingWeights.length][startingWeights[0].length];       

        for (int i = 0; i < weightGradient[0][0].length; i++)
        {          
            for (int j = 0; j < weightGradient[0].length; j++)
            {        
                double sumWeightGradient = 0;
                // Sums a weight from each calculated weight gradient from their 3D array position  
                for (int k = 0; k < weightGradient.length; k++)
                {
                    sumWeightGradient += weightGradient[k][j][i]; 
                }
                // Once summation has been found from past weights, use formula
                revisedWeights[j][i] = startingWeights[j][i] - (eta/2) * sumWeightGradient;         
            }
        }
        return revisedWeights;
    }

    public static void trainNetwork(int miniBatchSize, int currentBatch, int eta, double[][] activationLayer0, double[][] weightLayer1, double[][] weightLayer2, double[] biasLayer1, double[] biasLayer2)
    {
        for (int i = 0; i < miniBatchSize; i++)
        {              
            // Forward pass through Layer 1
            double[] zLayer1 = calculateZLayer(activationLayer0[(currentBatch * miniBatchSize) + i], weightLayer1, biasLayer1);
            double[] aLayer1 = calculateALayer(zLayer1);

            // Forward pass through Layer 2
            double[] zLayer2 = calculateZLayer(aLayer1, weightLayer2, biasLayer2);
            double[] aLayer2 = calculateALayer(zLayer2);

            // Get Cost
            double cost = calculateCost(aLayer2, classifcationSet[i]);

            // Backwards Propigation through Layer 2
            double[] gradiantBiasLayer2 = backwardPorpigationLayer2(aLayer2, classifcationSet[i]);
            //System.out.println("GradiantBias layer 2: " + Arrays.toString(gradiantBiasLayer2));
            double[][] weightGradientLayer2 = calculateWeightGradient(aLayer1, gradiantBiasLayer2);

            // Backwards Propigation through Layer 1
            double[] gradiantBiasLayer1 = backwardPorpigationLayer1(weightLayer2, gradiantBiasLayer2, aLayer1);
            //System.out.println("GradientBias layer 1: " + Arrays.toString(gradiantBiasLayer1));
            double[][] weightGradientLayer1 = calculateWeightGradient(activationLayer0[i], gradiantBiasLayer1);
            //System.out.println("Weight Gradient layer 1: " + Arrays.deepToString(weightGradientLayer1));

            calculatedWeightsLayer1[i] = weightGradientLayer1;
            calculatedWeightsLayer2[i] = weightGradientLayer2;

            calculatedBiasesLayer1[i] = gradiantBiasLayer1;
            calculatedBiasesLayer2[i] = gradiantBiasLayer2;
        }
    }

    public static void main(String args[])
    {
        // for number of batches in an epoch
        for (int currentBatch = 0; currentBatch < miniBatchPerEpoch; currentBatch++)
        {
            // TODO: Figure out a way to skip index in activationLayer for which miniBatch you are on
            trainNetwork(miniBatchSize, currentBatch, eta, activationLayer0, weightLayer1, weightLayer2, biasLayer1, biasLayer2);
                
            System.out.println("Memory of Weights Layer 1: \n" + Arrays.deepToString(calculatedWeightsLayer1).replaceAll("], ", "],\n") + "\n");
            System.out.println("Memory of Weights Layer 2: \n" + Arrays.deepToString(calculatedWeightsLayer2).replaceAll("], ", "],\n") + "\n");
            System.out.println("Memory of Bias Layer 1: \n" + Arrays.deepToString(calculatedBiasesLayer1).replaceAll("], ", "],\n") + "\n");
            System.out.println("Memory of Bias Layer 2: \n" + Arrays.deepToString(calculatedBiasesLayer2).replaceAll("], ", "],\n") + "\n");

            biasLayer1 = reviseBias(calculatedBiasesLayer1, biasLayer1);
            biasLayer2 = reviseBias(calculatedBiasesLayer2, biasLayer2);
            System.out.println("Updated Bias Layer 1: \n" + Arrays.toString(biasLayer1) + "\n");
            System.out.println("Updated Bias Layer 2: \n" + Arrays.toString(biasLayer2) + "\n");

            weightLayer1 = reviseWeights(calculatedWeightsLayer1, weightLayer1);
            weightLayer2 = reviseWeights(calculatedWeightsLayer2, weightLayer2);
            System.out.println("Updated Weight Layer 1: \n" + Arrays.deepToString(weightLayer1).replaceAll("], ", "],\n") + "\n");
            System.out.println("Updated Weight Layer 2: \n" + Arrays.deepToString(weightLayer2).replaceAll("], ", "],\n") + "\n");
        }
    }
}