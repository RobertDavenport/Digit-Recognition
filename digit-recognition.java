import java.lang.Math;
import java.util.Arrays;

/* Desc */
class DigitRecognition
{
    // Might not even need this....
    public static int miniBatchSize = 2;

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
{   {0, 1},
    {1, 0},
    {0, 1},
    {1, 0}
};

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
        for (int i = 0; i< gradientBias.length; i++)
        {
            for (int j = 0; j < aLayer.length; j++)
            {
                weightGradient[i][j] = aLayer[j] * gradientBias[i];
            }
        }
        return weightGradient;
    }

    //public static double[][] reviseWeights(double[][] weights)

    public static void main(String args[])
    {
        // 3D arrays for storing calculated weights for each run of a minibatch
        // TODO: might need to change size initialization on new training set
        double[][][] calculatedWeightsLayer1 = new double[miniBatchSize][weightLayer1.length][weightLayer1[0].length];
        double[][][] calculatedWeightsLayer2 = new double[miniBatchSize][weightLayer2.length][weightLayer2[0].length];
        // 3D arrays for storing calculated bias for each run of a minibatch
        double[][] calculatedBiasesLayer1 = new double[miniBatchSize][biasLayer1.length];
        double[][] calculatedBiasesLayer2 = new double[miniBatchSize][biasLayer2.length];
        for (int i = 0; i < miniBatchSize; i++)
        {   
            System.out.println("---===Run " + (i + 1) + "/" + miniBatchSize +" For MiniBatch===--- \n");
            // Forward pass through Layer 1
            double[] zLayer1 = calculateZLayer(activationLayer0[i], weightLayer1, biasLayer1);
            double[] aLayer1 = calculateALayer(zLayer1);
            //System.out.println("activation layer 1: " + Arrays.toString(aLayer1));

            // Forward pass through Layer 2
            double[] zLayer2 = calculateZLayer(aLayer1, weightLayer2, biasLayer2);
            double[] aLayer2 = calculateALayer(zLayer2);
            //System.out.println("activation layer 2: " + Arrays.toString(aLayer2));

            // Get Cost
            double cost = calculateCost(aLayer2, classifcationSet[i]);
            //System.out.println("cost: " + cost);

            // Backwards Propigation through Layer 2
            double[] gradiantBiasLayer2 = backwardPorpigationLayer2(aLayer2, classifcationSet[i]);
            //System.out.println("GradiantBias layer 2: " + Arrays.toString(gradiantBiasLayer2));
            double[][] weightGradientLayer2 = calculateWeightGradient(aLayer1, gradiantBiasLayer2);
            //System.out.println("Weight Gradient layer 2: " + Arrays.deepToString(weightGradientLayer2));

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
        
        System.out.println("Memory of Weights Layer 1: \n" + Arrays.deepToString(calculatedWeightsLayer1).replaceAll("], ", "],\n") + "\n");
        System.out.println("Memory of Weights Layer 2: \n" + Arrays.deepToString(calculatedWeightsLayer2).replaceAll("], ", "],\n") + "\n");
        System.out.println("Memory of Bias Layer 1: \n" + Arrays.deepToString(calculatedBiasesLayer1).replaceAll("], ", "],\n") + "\n");
        System.out.println("Memory of Bias Layer 2: \n" + Arrays.deepToString(calculatedBiasesLayer2).replaceAll("], ", "],\n") + "\n");
    }
}