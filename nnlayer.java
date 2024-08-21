import java.util.Scanner;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
/**
 * Builds an N Layer Network to solve problems like OR, AND, and XOR.
 * Can run based of set configurations from a .txt file and train with random weights or running with pre-set weights
 * It uses backpropagation to train the model. 
 * 1) It configures the parameters and variables to set the structure, mode, and values of the network
 * 2) Allocates proper memory for arrays depending on if running or training 
 * 3) Prints out the parameters for the user to see what it received 
 * 4) Fills in the arrays of the truth table and weights depending on testing or running and training type.
 * The network runs feedforward. From an input layer to a hidden layer to an output layer
 * 5) If running, it will find output value. If training, it will time the process and use backpropagation to train the network.
 * 6) Report the output results and saves the weights.
 *  
 *  
 *  Method Headers:
 *      public static void configParams(String fileName)
 *      public static void readControlFile()
 *      public static void echoConfigParams()
 *      public static void allocateMemory()
 *      public static void populateArrays()
 *      public static void fillTrainingWeights()
 *      public static void loadWeights()
 *      public static void saveWeights()
 *      public static void fillTruthTableTraining()
 *      public static void fillTruthTableRunning()
 *      public static double randomValue()
 *      public static void trainOrRun()
 *      public static void trainNetwork(int ti)
 *      public static void findNetworkAverageError()
 *      public static void findNetworkOutputRunning(int testCase)
 *      public static void findNetworkOutputTraining(int testCase)
 *      public static double activationFunction(double x)
 *      public static double sigmoidFunction(double x)
 *      public static double activationFunctionDeriv(double x)
 *      public static double sigmoidDeriv(double x)
 *      public static void reportResults()
 *      public static void main(String[] args)
 *      
 *  
 * @author Aeliya Grover
 * @version November 07 2023
 */
public class nnlayer
{
   private static boolean training, override;
   
   private static String weightPathway, filePathway, controlFilePathway;
   
   private static double randomLow, randomHigh, byteRead;

   private static int  numLayers, numTrainingCases, maxLength, lastLayer, firstLayer, lastJ;
   private static int[] beta;
   
   private static int  maxIterations, iterationsReached, printIterationNum;
   
   private static double[] upperCasePsiJ;
   private static double[][] a, theta;     
   private static double[][][] w;
      
   private static double lambda, errorThreshold, totalError, currentAverageError, partialDeriv;
   private static double upperCaseOmegaJ, upperCaseOmegaK;
   private static double thetaAB, upperCasePsiK, lowerCaseOmega;
   private static double[][] psi; 
   
   private static double[][] truthTable, t;  
   
   private static double startTime, endTime, totalTime;

   private static File controlFile; 

   /**
    * Sets variables to help construct the N-layer structure, parameters, and mode.
    * Has a preset file pathway, but can override it if another directory is passed at runtime
    */
   public static void configParams(String fileName) throws Exception
   {   
      controlFilePathway = "/Users/24aeliyag/Desktop/NeuralNetworks/ControlFile.txt";  //default
      
      maxLength = 0; // to find the largest layer length
      
      
      if (fileName.length() > 0)
      {
         controlFilePathway = fileName;
      }
      
      controlFile = new File (controlFilePathway);
      
      readControlFile();  
      
      lastLayer = numLayers - 1;
      firstLayer = 0;
      lastJ = 2;   
      
      iterationsReached = 0;
   } // public static void configParams(String fileName)
   
   /**
    * Uses values from the control file to fill in network structure, parameter, and mode.
    * Finds weight file pathway
    * The .txt file must match up to the specific format in order for values to be transfered correctly.
    */
   public static void readControlFile() throws Exception 
   {
      Scanner controlScan = new Scanner(controlFile); 
      
      weightPathway = controlScan.nextLine();
      
      training = controlScan.nextBoolean();
      randomLow = controlScan.nextDouble(); 
      randomHigh = controlScan.nextDouble(); 
      numLayers = controlScan.nextInt(); 
      
      for (int n = 0; n < numLayers; n++)
      {
         controlScan.nextInt();     // skips through layer construct
      }
      
      numTrainingCases = controlScan.nextInt();
      
      lambda = controlScan.nextDouble();
      maxIterations = controlScan.nextInt();
      errorThreshold = controlScan.nextDouble();
      printIterationNum = controlScan.nextInt();
      
   } // public static void readControlFile()
   
   /**
    * Prints out the variables to user, informing of network configuration.
    * If training, it also outputs training parameters
    */
   public static void echoConfigParams()
   {
      System.out.println("---------------------------------------------------------");
      System.out.println("ECHOING CONFIGURATION PARAMETERS");
      System.out.println();
      System.out.print("    Network Configuration = " + beta[firstLayer]); 
      
      for (int n = 1; n < numLayers; n++)
      {
         System.out.print("-" + beta[n]);
      }
     
      if (training)    // Only if training:
      {
         System.out.println("    Runtime Trainng Parameters: ");                                                                           
         System.out.println("        Random number range: (" + randomLow + "," + randomHigh + ")");         
         System.out.println("        Max iterations: " + maxIterations);                           
         System.out.println("        Error threshold: " + errorThreshold);                              
         System.out.println("        Lambda value: " + lambda);                               
        
      } // if (training) 
   } // public static void echoConfigParams()
   
   /**
    * Defining and dimensioning arrays dependent on whether the program is training or running
    * Requires the .txt control file to be in the proper format
    */
   public static void allocateMemory() throws Exception
   {
      Scanner controlScan = new Scanner(controlFile); 
      
      controlScan.nextLine();

      controlScan.nextBoolean();
      controlScan.nextDouble(); 
      controlScan.nextDouble(); 
      
      controlScan.nextInt();
      
      beta = new int[numLayers];
      
      for (int n = 0; n < numLayers; n++)
      {
         beta[n] = controlScan.nextInt(); 
         
         if (beta[n] > maxLength)
         {
            maxLength = beta[n];
         }
         
      } // for (int n = 0; n < numLayers; n++)
      
      a = new double[numLayers][maxLength];

      truthTable = new double[numTrainingCases][beta[firstLayer]];
      w = new double[numLayers][maxLength][maxLength];

      if (training)
      {
         t = new double[numTrainingCases][beta[lastLayer]];               // expected output
         psi = new double[numLayers][maxLength];
         theta = new double[numLayers][maxLength];
      } // if (training)
   } // public static void allocateMemory()
   
   /**
    * Assigns values for necessary arrays dependent on if training or running. 
    * Fills in the truth table arrays with proper values for inputs. If training, fills out the expected outputs.
    * If the network is training, it will find random weight values and fill the expected outputs.
    * If the network is running, it requires weight values from the user through a .txt file. 
    */
   public static void populateArrays() throws Exception
   { 
      if (training)                               // Fills in the weights dependent on training mode
      {
         fillTruthTableTraining();
         fillTrainingWeights();
      } // if (training)
      else
      {
         fillTruthTableRunning();
         loadWeights();
      } // if (training)... else
   } // public static void populateArrays()
  
   /*
    * Generates weights using a random number generator to start training the network.
    */
   public static void fillTrainingWeights()
   {
      for (int a = 0; a < lastLayer; a++)
      {
         
         for (int b = 0; b < beta[a]; b++)
         {
           
            for (int y = 0; y < beta[a + 1]; y++)
            {
               w[a][b][y] = randomValue();
            }
        
         } // for (int b = 0; b < beta[a]; b++)
         
      } // for (int a = 0; a < lastLayer; a++)
      
   } // public static void fillTrainingWeights()
   
   /*
    * Sets the weights for the network by reading a txt file. 
    * The order of the weights file must be constant to match the method.
    * It will then fill the weights that connect the hidden layer to the output layer. 
    */
   public static void loadWeights() throws Exception
   {
      File file = new File(weightPathway); 
      Scanner scan = new Scanner(file); 
      
      for (int a = 0; a < lastLayer; a++)
      {
        
         for (int b = 0; b < beta[a]; b++)
         {
           
            for (int y = 0; y < beta[a+1]; y++)
            {
               w[a][b][y] = scan.nextDouble();
            }
            
         } // for (int b = 0; b < beta[a]; b++)
         
      } // for (int a = 0; a < lastLayer; a++)
      
   } // public static void loadWeights()

   /**
    * Saves the weights for the network to a .txt file.
    * It first saves the weights connecting the input layer to the hidden layer.
    * After, it saves the weights connecting the hidden layer to output layer.
    */
   public static void saveWeights()
   {
      try
      {
         FileWriter fill = new FileWriter(weightPathway);
         
         for (int a = 0; a < lastLayer; a++)
         {
            
            for (int b = 0; b < beta[a]; b++)
            {
               
               for (int y = 0; y < beta[a + 1]; y++)
               {
                  fill.write(w[a][b][y] + " ");
               }
               
            } // for (int b = 0; b < beta[a]; b++)
            
         } // for (int a = 0; a < lastLayer; a++)
        
         fill.close();
         
      } // try 
      catch (IOException error)
      {
         error.printStackTrace();
      } // try...catch
   } // public static void saveWeights()
  
   /**
    * Sets the values for the truth table and expected outputs to train the network on.
    * The .txt file must follow the proper format for values to transfer correctly
    */
   public static void fillTruthTableTraining() throws Exception
   {
      Scanner controlScan = new Scanner(controlFile); 
      
      controlScan.nextLine();
      
      controlScan.nextBoolean();         // Skips through network parameters to get to truth table
      controlScan.nextDouble(); 
      controlScan.nextDouble(); 
      
      controlScan.nextInt();             // Number of layers
      
      for (int n = 0; n < numLayers; n++)
      {
         controlScan.nextInt();          // Each layer length
      }

      controlScan.nextInt();             // Number of training cases
      
      controlScan.nextDouble();
      controlScan.nextInt();
      controlScan.nextDouble();
      controlScan.nextInt();
      

      for (int trainC = 0; trainC < numTrainingCases; trainC++)  
      {
         String inputFile = controlScan.next();
   
         File numbersRead = new File(inputFile);
           try ( Scanner inputStream = new Scanner(numbersRead); ) {
           //   System.out.println(inputFile);
          
            
            for (int i = 0; i < beta[firstLayer]; i++) {
           
               byteRead = (double) inputStream.nextDouble();
      
               truthTable[trainC][i] = byteRead; 
             //  System.out.println(truthTable[trainC][i]);
      
            }
           }  
        
    
         
         for (int ii = 0; ii < beta[lastLayer]; ii++)
         {
            t[trainC][ii] = controlScan.nextDouble();                   // Saves expected values to train on
        
         //   System.out.println(t[trainC][ii]);
         }   
         
      } // for (int trainC = 0; trainC < numTrainingCases; trainC++)
      
      
   } // public static void fillTruthTableTraining()
   
   /**
    * Sets the values for the truth table the network runs on by reading a .txt file
    * The file must be in the proper format for values to read properly
    */
   public static void fillTruthTableRunning() throws Exception
   {
      Scanner controlScan = new Scanner(controlFile); 
      
      controlScan.nextLine();
      controlScan.nextBoolean();         // Skips through network parameters to get to truth table
      controlScan.nextDouble(); 
      controlScan.nextDouble(); 
   
      controlScan.nextInt();             // number of layers
      
      for (int n = 0; n < numLayers; n++)
      {
         controlScan.nextInt();          // Each layer length
      }

      controlScan.nextInt();             
      
      
      controlScan.nextDouble();
      controlScan.nextInt();
      controlScan.nextDouble();
      controlScan.nextInt();
      
      for (int trainC = 0; trainC < numTrainingCases; trainC++)
      {
         
         String inputFile = controlScan.next();
         

         File numbersRead = new File(inputFile);
           try ( Scanner inputStream = new Scanner(numbersRead); ) {
        
          
          for (int i = 0; i < beta[firstLayer]; i++) {
             byteRead = inputStream.nextDouble();
            //System.out.println(byteRead);
             
    
             truthTable[trainC][i] = byteRead; 
            // System.out.println(truthTable[trainC][i]);
    
          }
         }  
      
      
  
         
      } // for (int trainC = 0; trainC < numTrainingCases; trainC++)
   } // public static void fillTruthTableRunning() 

   /**
    * Randomly generates a value based on a preset range.
    * Returns the randomly generated value.
    */
   public static double randomValue()
   {
      return (double)(Math.random() * (randomHigh - randomLow)) + randomLow;    
   }
   
   /**
    * If the training variable is true, it will run and train the feedforward N layer Network.
    * Backpropagation will be used and the training will stop if an error threshold or maximum iteration is met.
    * If training, it also saves training time.
    * If training is false, it will find the output based on predetermined weights.
    */
   public static void trainOrRun() throws Exception
   {
      if (!training) 
      {
         
         for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)
         {
            findNetworkOutputRunning(numTraining);                 // Running the network on inputed weights
         }
         
      } // if (!training) 
      else
      {
         startTime = System.currentTimeMillis();
         System.out.println("------------------------------------------");
         System.out.println("Training Network: ");
         currentAverageError = errorThreshold + errorThreshold;
        
         while (currentAverageError > errorThreshold && iterationsReached < maxIterations)       // Determines if another round of backpropagation needs to be found
         {
            
            for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)             // Loops through each training case
            {
               findNetworkOutputTraining(numTraining);
               trainNetwork();     
            } // for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)
             
            if (iterationsReached % printIterationNum == 0.0)
            {
               System.out.println();    
               System.out.println("   Error: " + currentAverageError );
               System.out.println("   Training Iteration: " + iterationsReached);
            }
            findNetworkAverageError();  
            iterationsReached++;  
         } // while (currentError > errorThreshold && iterationsReached < maxIterations) 
         
         endTime = System.currentTimeMillis();
         totalTime = endTime - startTime;
      } // if (!training)... else 
      saveWeights();
   } // public static void trainOrRun()
   
   /**
    * Finds the deltaW for the weights and makes changes
    * Requires saved values from when running through the network.
    */
   public static void trainNetwork()
   {
      for (int n = numLayers - lastJ; n > 1; n--)  // Generalized training, avoids first layer 
      {
         
         for (int j = 0; j < beta[n]; j++)        
         {
            upperCaseOmegaJ = 0.0;
            
            for (int ii = 0; ii < beta[n + 1]; ii++)
            {
               upperCaseOmegaJ += psi[n + 1][ii] * w[n][j][ii];
               w[n][j][ii] += lambda * a[n][j] * psi[n + 1][ii];
            }
            
            psi[n][j] = upperCaseOmegaJ * activationFunctionDeriv(theta[n][j]);

         } // for (int n = numLayers - lastJ; n > 1; n--)
         
      } // for (int n = numLayers - 2; n > 1; n--)
      
      for (int k = 0; k < beta[1]; k++)            // Training the first layers
      {
         upperCaseOmegaK = 0.0; 
         
         for (int j = 0; j < beta[lastJ]; j++)
         {
            upperCaseOmegaK += psi[lastJ][j] * w[1][k][j];
            w[1][k][j] += lambda * a[1][k] * psi[lastJ][j];
         }

         upperCasePsiK = upperCaseOmegaK * activationFunctionDeriv(theta[1][k]);
         
         for (int m = 0; m < beta[0]; m++)
         {
            w[0][m][k] += lambda * a[0][m] * upperCasePsiK;
         }
      } // for (int k = 0; k < beta[1]; k++) 
      
   } // public static void trainNetwork(int ti)

   /**
    * Runs through each test case to find the total error and average error for the network 
    */
   public static void findNetworkAverageError()
   {
         totalError = 0.0; 
         
         for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)
         {
            findNetworkOutputRunning(numTraining);

            for (int ii = 0; ii < beta[lastLayer]; ii++)
            {
               lowerCaseOmega = a[lastLayer][ii] - t[numTraining][ii];
              
               totalError += lowerCaseOmega * lowerCaseOmega; 
            }  

         } // for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)
         
         totalError *= 0.5;
         currentAverageError = totalError / (double) numTrainingCases;
   }// public static void findNetworkAverageError()
   
   
   /**
     * Takes in a test case to find the proper inputs. 
     * Then, solves the hidden layer values and output value.
     * Solves using feedforward passage. 
     */
   public static void findNetworkOutputRunning(int testCase)
   {
      
      for (int m = 0; m < beta[firstLayer]; m++)    // Set up
      {
         a[firstLayer][m] = truthTable[testCase][m];         // fills in input layer Am
      }
      
      
      for (int aa = 1; aa < numLayers; aa++)
      {
         
         for (int b = 0; b < beta[aa]; b++)
         {
            
            thetaAB = 0.0; 
            
            for (int y = 0; y < beta[aa - 1]; y++)
            {
               thetaAB += a[aa - 1][y] * w[aa - 1][y][b];
            }
           
            a[aa][b] = activationFunction(thetaAB);
           
         } // for (int b = 0; b < beta[aa]; b++)
         
      } // for (int aa = 1; aa < numLayers; aa++)

   } // public static void findNetworkOutputRunning(int testCase)
   
   /**
    * Takes in a test case to find the proper inputs. 
    * Then, solves the hidden layer values and output value.
    * Saves the values to be used in backpropagation.
    */
   public static void findNetworkOutputTraining(int testCase)
   {
      for (int m = 0; m < beta[firstLayer]; m++)    // Set up
      {
         a[firstLayer][m] = truthTable[testCase][m];         // fills in input layer Am
      }
      
      for (int aa = 1; aa < numLayers - 1; aa++)    // Generalized for n-layers, avoids last layer
      {
         
         for (int b = 0; b < beta[aa]; b++)
         {
            
            theta[aa][b] = 0.0; 
            
            for (int y = 0; y < beta[aa - 1]; y++)
            {
               theta[aa][b] += a[aa - 1][y] * w[aa - 1][y][b];
            }
            
            a[aa][b] = activationFunction(theta[aa][b]);
            
         } // for (int b = 0; b < beta[aa]; b++)
         
      } // for (int aa = 1; aa < numLayers; aa++)
      
      
      for (int b = 0; b < beta[lastLayer]; b++)   // Specific for last layers
      {
         
         theta[lastLayer][b] = 0.0; 
         
         for (int y = 0; y < beta[lastLayer - 1]; y++)
         {
            theta[lastLayer][b] += a[lastLayer - 1][y] * w[lastLayer - 1][y][b];
         }
         
         a[lastLayer][b] = activationFunction(theta[lastLayer][b]);
         psi[lastLayer][b] = (t[testCase][b] - a[lastLayer][b]) *  activationFunctionDeriv(theta[lastLayer][b]);
         
      } // for (int b = 0; b < beta[lastLayer]; b++)
      
   } // public static void findNetworkOutputTraining(int testCase)

   /**
    * Passes the input, x, through the selected threshold function and returns the computed value to the network
    */
   public static double activationFunction(double x)
   {
      return sigmoidFunction(x);
   }
   
   /**
    * Runs the sigmoid function if selected as the activation function 
    * Passes the parameter x through the function and returns the output value
    */
   public static double sigmoidFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }
   public static double hyperbolic (double x)
   {
      return (1.0 - Math.exp(-x)) / (1.0 + Math.exp(-x));
   }
   public static double hyperbolicDeriv(double x)
   {
     double top =  (2.0 * Math.exp(x));
     double bottom =  (1 + Math.exp(x));
     double square = bottom * bottom; 
     return top / square;
   }
   
   
   /**
    * Passes the input through the derivative of the selected activation function   
    * Passes the parameter x through the function and returns it for the network
    */
   public static double activationFunctionDeriv(double x)
   {
      return sigmoidDeriv(x);
   } 
   
   /**
    * Stores the function for the sigmoid derivative to be used for activationFunctionDeriv
    * Passes the paramter x through the function and returns it to be used in error analysis.
    */
   public static double sigmoidDeriv(double x)
   {
      double activationFunctionResult = activationFunction(x);
      return activationFunctionResult * (1.0 - activationFunctionResult);
   }
    
      
   /**
    * Prints out a truth table with output values found by the n-layer network 
    * If training, provides explanation as to why training stopped with end error, iteration count, and time taken to train. 
    * Calls to save current weights to .txt file.
    */
   public static void reportResults() throws Exception
   {
      System.out.println();
      System.out.println("--------------------------------------------------------------------");
      System.out.println("Reporting Results:");
      System.out.println();
        
      if (training)
      {
         System.out.println("   Training Exit Information:");              
         
         if (currentAverageError <= errorThreshold)                                         // Determines if stopped by reaching the error threshold
         {
            System.out.println("      Stopped because network reached error threshold");
         }
         
         if (iterationsReached >= maxIterations)                                     // Determines if stopped by reaching the max iterations
         {
            System.out.println("      Stopped because network reached max iterations");
         }
         
         System.out.println();
         System.out.println("   Total Error Reached: " + currentAverageError); 
         System.out.println("   Total iterations reached: " + iterationsReached);
         System.out.println("   Total time to train (ms): " + totalTime);
      } // if (training)
      
      System.out.println();
      System.out.println("    ------------------");
      
      for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)
      {
         System.out.print("    |");
         for (int m = 0 ; m < beta[firstLayer]; m++)
         {
        //   System.out.print(" " + truthTable[numTraining][m] + " |");    // Prints out the truth table
             
         }
        
         findNetworkOutputRunning(numTraining);  
         
         for (int ii = 0; ii < beta[lastLayer]; ii++)
         {
            System.out.print("F" + ii + ": " + a[lastLayer][ii] + "    ");                  // Prints out the output values
         }
                  
         System.out.println();              
      } // for (int numTraining = 0; numTraining < numTrainingCases; numTraining++)

      System.out.println("    --------------------");

      saveWeights();
   } // public static void reportResults()
   
   /**
    * Reading from a .txt file, the program starts by setting variables and allocating proper space. 
    * Then, continues by populating the arrays. 
    * The network either runs or trains using backpropagation based on a set variable.
    * A report is then given summarizing the run. 
    */
   public static void main(String[] args) throws Exception
   {      
      if (args.length > 0)
      {
         configParams(args[0]);
      }
      else
      {
         configParams("");
      }
      allocateMemory();
      echoConfigParams();
      populateArrays();
      trainOrRun();
      reportResults();
      
   } // public static void main(String[] args)
} // public class nlayer