package classes;

import classes.dataHelper.AnomalyDataSetIterator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class AnomalyDetection {

    private static int traininSetBatchSize = 64;
    private static int testSetBatchSize =1;
    private static int numEpochs = 30;
    private static final int seed = 123;
    private static File modelFilename = new File(System.getProperty("user.dir"),"anomalyDetectorModel.zip");
    private static MultiLayerNetwork model;

    public static void main(String[] args) throws IOException {

        String trainingSetData = new ClassPathResource("train.csv").getFile().getAbsolutePath();
        String testSetData = new ClassPathResource("test.csv").getFile().getAbsolutePath();

        DataSetIterator trainingSetIter = new AnomalyDataSetIterator(trainingSetData, traininSetBatchSize);
        DataSetIterator testSetIter = new AnomalyDataSetIterator(testSetData, testSetBatchSize);

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(trainingSetIter);

        trainingSetIter.setPreProcessor(normalizer);
        testSetIter.setPreProcessor(normalizer);

        if (modelFilename.exists()){

            Nd4j.getRandom().setSeed(seed);
            System.out.println("loading neural network...");
            model = ModelSerializer.restoreMultiLayerNetwork(modelFilename);

        }
        else{
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam())
                    .l2(1e-4)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.TANH)
                    .list()
                    .layer(new LSTM.Builder().name("encoder_0").nIn(trainingSetIter.inputColumns()).nOut(100).build())
                    .layer(new LSTM.Builder().name("encoder_1").nOut(80).build())
                    .layer(new LSTM.Builder().name("encoder_2").nOut(5).build())
                    .layer(new LSTM.Builder().name("decoder_1").nOut(80).build())
                    .layer(new LSTM.Builder().name("decoder_2").nOut(100).build())
                    .layer(new RnnOutputLayer.Builder().name("output").nOut(trainingSetIter.totalOutcomes())
                            .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
                    .build();

            model = new MultiLayerNetwork(conf);
            model.init();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
            uiServer.attach(statsStorage);

            // training
            model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            model.fit(trainingSetIter, numEpochs);

            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved");

        }


        int totalScore = 0;
        while(testSetIter.hasNext()){
             DataSet testDataSet  = testSetIter.next();
             double score = model.score(testDataSet);
             totalScore += score;
        }

        testSetIter.reset();
        int idx = 0;
        double threshold = 10.0;
        ArrayList idxList = new ArrayList<>();
        System.out.println("Anomaly Detected...");
        System.out.println("Finding data with anomaly");
        while(testSetIter.hasNext()){
            DataSet testDataSet  = testSetIter.next();
            double score = model.score(testDataSet);
            normalizer.revert(testDataSet);
            if (score > threshold){
                System.out.println("idx " + idx + testDataSet.getFeatures().toString() + score);
                idxList.add(idx);
            }
            idx ++;
        }

        System.out.println("Index of data with anomaly found: ");
        System.out.println(idxList);

    }

}
