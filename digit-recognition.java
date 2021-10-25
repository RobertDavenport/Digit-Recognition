import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;
import java.io.*;
import java.util.stream.DoubleStream;

/* Desc */
class DigitRecognition
{
    // Network Parameters
    public static int traningSetSize = 60000;
    public static int activationLayerInputSize = 784;
    public static int nodesInLayer1 = 30;
    public static int nodesInLayer2 = 10;

    // Statistical Tracking
    public static List<Integer> correctList = new ArrayList<>();
    public static List<Integer> incorrectList = new ArrayList<>();

    // Input Layer
    public static double[][] activationLayer0 = new double[traningSetSize][activationLayerInputSize];

    // Weights
    public static double[][] weightLayer1 = new double[nodesInLayer1][activationLayerInputSize];
    public static double[][] weightLayer2 = new double[nodesInLayer2][nodesInLayer1];

    // Bias
    public static double[] biasLayer1 = new double[nodesInLayer1];
    public static double[] biasLayer2 = new double[nodesInLayer2];

    // Classifications
    public static double[][] classifcationSet = new double[traningSetSize][nodesInLayer2];

    public static int miniBatchSize = 10;
    public static int eta = 3; // learning rate
    public static int miniBatchPerEpoch = (activationLayer0.length / miniBatchSize);
    public static int totalEpochs = 30;

    // 3D arrays for storing calculated weights for each run of a minibatch
    public static double[][][] calculatedWeightsLayer1 = new double[miniBatchSize][weightLayer1.length][weightLayer1[0].length];
    public static double[][][] calculatedWeightsLayer2 = new double[miniBatchSize][weightLayer2.length][weightLayer2[0].length];
    // 2D arrays for storing calculated bias for each run of a minibatch
    public static double[][] calculatedBiasesLayer1 = new double[miniBatchSize][biasLayer1.length];
    public static double[][] calculatedBiasesLayer2 = new double[miniBatchSize][biasLayer2.length];
 
    public static double[] calculateZLayer(double[] activationLayer, double[][] weightLayer, double[] biasLayer) {
        double[] zLayer = new double[weightLayer.length];       
        for (int i = 0; i < weightLayer.length; i++) {
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
        for (int i = 0; i < zLayer.length; i++)
        {
            aLayer[i] = 1.0 / (1.0 + Math.pow(Math.E, -zLayer[i]));
        }
        return aLayer;
    }

    // unused in network training, judges how inaccurate the network was
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

    // uses the stored biases to calculate the new bias
    public static double[] reviseBias(double[][] biasGradient, double[] startingBias)
    {
        double[] revisedBias = new double[startingBias.length];       

        for (int i = 0; i < startingBias.length; i++)
        {
            double sumBiasGradient = 0;
            // sum the biases for each run in the minibatch
            for (int j = 0; j < biasGradient.length; j++)
            {
                sumBiasGradient += biasGradient[j][i]; 
            }
            revisedBias[i] = startingBias[i] - ((double)eta/miniBatchSize) * sumBiasGradient;         
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
                revisedWeights[j][i] = startingWeights[j][i] - ((double)eta/miniBatchSize) * sumWeightGradient;         
            }
        }
        return revisedWeights;
    }

    // returns the index of the largest element in an array
    public static int getMaxArrayElementIndex(double[] array){
        int max = 0;
        for (int i = 0; i < array.length; i++){           
            if (array[i] > array[max]){
                max = i;
            }
        }
        return max;
    }

    public static void compareClassification(double[] output, double[] classification){
        int maxOutput = getMaxArrayElementIndex(output);
        int maxClassification = getMaxArrayElementIndex(classification);
        // if our indexes match, then the output was correct
        if (maxOutput == maxClassification){
            correctList.add(maxOutput);
        }
        else{
            incorrectList.add(maxOutput);
        }
    }

    public static void trainNetwork(int miniBatchSize, int currentBatch, int eta, double[][] activationLayer0, double[][] weightLayer1, double[][] weightLayer2, double[] biasLayer1, double[] biasLayer2)
    {
        // for each activation in a minibatch
        for (int i = 0; i < miniBatchSize; i++)
        {              
            // Forward pass through Layer 1
            double[] zLayer1 = calculateZLayer(activationLayer0[(currentBatch * miniBatchSize) + i], weightLayer1, biasLayer1);
            double[] aLayer1 = calculateALayer(zLayer1);

            // Forward pass through Layer 2
            double[] zLayer2 = calculateZLayer(aLayer1, weightLayer2, biasLayer2);
            double[] aLayer2 = calculateALayer(zLayer2);

            // Determine if network output matches classification
            compareClassification(aLayer2, classifcationSet[(currentBatch * miniBatchSize) + i]);

            // Get Cost
            double cost = calculateCost(aLayer2, classifcationSet[(currentBatch * miniBatchSize) + i]);

            // Backwards Propigation through Layer 2
            double[] gradiantBiasLayer2 = backwardPorpigationLayer2(aLayer2, classifcationSet[(currentBatch * miniBatchSize) + i]);
            double[][] weightGradientLayer2 = calculateWeightGradient(aLayer1, gradiantBiasLayer2);

            // Backwards Propigation through Layer 1
            double[] gradiantBiasLayer1 = backwardPorpigationLayer1(weightLayer2, gradiantBiasLayer2, aLayer1);
            double[][] weightGradientLayer1 = calculateWeightGradient(activationLayer0[(currentBatch * miniBatchSize) + i], gradiantBiasLayer1);

            calculatedWeightsLayer1[i] = weightGradientLayer1;
            calculatedWeightsLayer2[i] = weightGradientLayer2;

            calculatedBiasesLayer1[i] = gradiantBiasLayer1;
            calculatedBiasesLayer2[i] = gradiantBiasLayer2;
        }
    }

    // Now unused, must be 10 nodes in output layer to take the max. Leaving for future exploration of a 4 node output
    // Takes a String value and maps its 10-digit binary representation into an array.
    private static double[] stringToBinaryArray(String x){
        // convert to Integer and turn to Binary... probably a better way than going from string -> int -> string
        String binary = Integer.toBinaryString(Integer.parseInt(x));
        // Format to 10-digit Binary number
        binary = String.format("%4s", binary).replaceAll(" ", "0");
        // convert String into Array and cast each element to a double... might change to integers in future
        double[] binaryArray = Arrays.stream(binary.split(""))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
        return binaryArray;
    }

        // Takes a String value of classifier and converts it to our desired classification output.
        // Example: digit "5" should be formatted to [0,0,0,0,0,1,0,0,0,0] 
        private static double[] formatClassifications(String x){
            // convert to Integer
            int digit = Integer.parseInt(x);
            // add the digit number of leading zeros
            String leadingZeros = new String(new char[digit]).replace("\0", "0");
            // since we have 10 output nodes, this is 10 minus the leading zeros. Additional subtraction for the classifier digit
            String trailingZeros = new String(new char[(10-(digit + 1))]).replace("\0", "0");
            String formattedClassification = leadingZeros + "1" + trailingZeros;
            // convert from string to double array
            double[] classificationArray = Arrays.stream(formattedClassification.split(""))
                                .mapToDouble(Double::parseDouble)
                                .toArray();
            return classificationArray;
        }

    private static void fetchData() {
        // Get data set by csv entered
        try (BufferedReader br = new BufferedReader(new FileReader("mnist_train.csv"))) {
            String line;
            int i = 0;          
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                for (int j = 0; j < values.length; j++) { 
                    // first element in the line is our classification
                    if (j == 0) {
                        // format digit into an an array
                        classifcationSet[i] = formatClassifications(values[j]);
                    }                                
                }
                // create new array removing first element and coverting to double          
                double[] array = Arrays.stream(Arrays.copyOfRange(values, 1, values.length)).mapToDouble(Double::valueOf).toArray();
                // normalize input between 0 and 1
                activationLayer0[i] = DoubleStream.of(array).map(p->p/255).toArray();
                i++;
            }      
        }
        catch(IOException ie){ }
    }

    // This is as a dynamic an approach as I could come up with... Expects a csv file with the propper format.
    // Since the weights are 2D-Arrays, it must keep track of the the current CSV row and subtract to begin inserting at 0
    private static void loadNueralNetwork() throws IOException {
        // Get data set by csv entered
        try (BufferedReader br = new BufferedReader(new FileReader("pretrained.csv"))) {
            String line;
            int csvRow = 0;          
            while ((line = br.readLine()) != null) {
                // creates an array from a line in the csv file
                String[] values = line.split(",");
                // for this section, csvRow should align exactly with index
                if (csvRow < weightLayer1.length) {
                    weightLayer1[csvRow] = Arrays.stream(values).mapToDouble(Double::valueOf).toArray();
                }
                // subtract out the previous csvRow count to "reset" index to 0
                else if (csvRow < weightLayer1.length + weightLayer2.length ) {
                    weightLayer2[csvRow - weightLayer1.length] = Arrays.stream(values).mapToDouble(Double::valueOf).toArray();
                }
                // + 1 since the biasLayer is a 1D array
                else if (csvRow < weightLayer1.length + weightLayer2.length + 1) {
                    biasLayer1 = Arrays.stream(values).mapToDouble(Double::valueOf).toArray();
                }
                else {
                    biasLayer2 = Arrays.stream(values).mapToDouble(Double::valueOf).toArray();
                }
                csvRow++;                                
            }      
        }
    }

    // returns a random array with values between -1 and 1
    public static double[][] randomizeWeights(double[][] weights){
        for (int i = 0; i < weights.length; i++){
            weights[i] = ThreadLocalRandom.current().doubles(weights[0].length, -1, 1).toArray();
        }
        return weights;
    }

    // returns a random double between -1 and 1
    public static double[] randomizeBias(double[] bias){
        return ThreadLocalRandom.current().doubles(bias.length, -1, 1).toArray();
    }

    public static void trainNewNueralNetwork(){

        // initialize network with random weights and biases
        weightLayer1 = randomizeWeights(weightLayer1);
        weightLayer2 = randomizeWeights(weightLayer2);
        biasLayer1 = randomizeBias(biasLayer1);
        biasLayer2 = randomizeBias(biasLayer2);
        // get the training set for inputs and classifications
        fetchData();

        // an Epoch
        for (int currentEpoch = 0; currentEpoch < totalEpochs; currentEpoch++)
        {
            // clear statistics from last epoch
            correctList.clear();
            incorrectList.clear();

            System.out.println("\n---===Starting Epoch " + (currentEpoch + 1) + "===---\n");
            
            // a Minibatch
            for (int currentBatch = 0; currentBatch < miniBatchPerEpoch; currentBatch++)
            {
                trainNetwork(miniBatchSize, currentBatch, eta, activationLayer0, weightLayer1, weightLayer2, biasLayer1, biasLayer2);
                
                // revise Bias and Weights after a minibatch
                biasLayer1 = reviseBias(calculatedBiasesLayer1, biasLayer1);
                biasLayer2 = reviseBias(calculatedBiasesLayer2, biasLayer2);

                weightLayer1 = reviseWeights(calculatedWeightsLayer1, weightLayer1);
                weightLayer2 = reviseWeights(calculatedWeightsLayer2, weightLayer2);
            }
            
            // Print Statistics
            int correctTotal = 0;
            int completeTotal = 0;
            for (int i = 0; i < 10; i++)
            {
                int correct = Collections.frequency(correctList, i);
                int total = correct + Collections.frequency(incorrectList, i);
                correctTotal += correct;
                completeTotal += total;
                System.out.print(i + " = " + String.valueOf(correct) + "/" + String.valueOf(total) + "\t");
            }

            System.out.print("Accuracy = " + String.valueOf(correctTotal) + "/"+ String.valueOf(completeTotal) + " " + String.format("%.2f", ((double)correctTotal / completeTotal) * 100) + "% \n");
        }
    }

    public static void saveNueralNetwork() throws IOException {
        BufferedWriter br = new BufferedWriter(new FileWriter("pretrained.csv"));
        StringBuilder sb = new StringBuilder();

        for (double[] array : weightLayer1) {
            for (double element : array) {
                sb.append(element);
                sb.append(",");
            }
            sb.append("\n");
        }

        for (double[] array : weightLayer2) {
            for (double element : array) {
                sb.append(element);
                sb.append(",");
            }
            sb.append("\n");
        }

        for (double element : biasLayer1) {
            sb.append(element);
            sb.append(",");
        }
        sb.append("\n");

        for (double element : biasLayer2) {
            sb.append(element);
            sb.append(",");
        }
        br.write(sb.toString());
        br.close();
    }

    public static void printMainMenu(){
        System.out.println("[0] Train New Nueral Network");
        System.out.println("[1] Save Nueral Network");
        System.out.println("[2] Load Existing Nueral Network");
        System.out.println("[99] Exit Application\n");
        System.out.println("Please Enter Your Selection..  ");
    }

    public static int getUserSelection(){
        Scanner scanner = new Scanner(System.in);
        int input = scanner.nextInt();
        return input;
    }

    public static void main(String args[]) throws IOException
    {
        int userInput = -1;
        // print Menu options to console
        printMainMenu();
        // respond according to selection
        while(userInput != 99){
            userInput = getUserSelection();
            switch (userInput){
                case 0: System.out.println("Training The Nueral Network\n");
                        trainNewNueralNetwork();
                        break;

                case 1: System.out.println("Saving The Nueral Network\n");
                        saveNueralNetwork();
                        break;
                case 2: System.out.println("Loading a Pretrained Nueral Network\n");
                        loadNueralNetwork();
                        break;
                case 99: System.out.println("Exiting Application\n");
                        break;
                default: System.out.println("Please choose one of the menu options\n");
                break;
            }
            printMainMenu();
        }        
    }
}