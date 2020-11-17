import java.util.ArrayList;

public class RadialBasisFunctionNetwork {

    private final int BASIC_FUNCTION_TYPE1 = 1;
    private final int BASIC_FUNCTION_TYPE2 = 2;
    private final int BASIC_FUNCTION_TYPE3 = 3;

    private final int ADJUST_W_THETA = -1;
    private final int ADJUST_ALL = -2;

    private int P;
    private int J; // num of neuron
    private int N; // num of data
    private int cycle = 0;
    private int maxLearningCycle;

    private double offset; // ???
    private double[][] x; // 1 ~ N, 1 ~ P
    private double[][] m;
    private double[] sigma;
    private double learningRate;

    private ArrayList<Double> E = new ArrayList<>();
    private ArrayList<double[]> w = new ArrayList<>(); // 0 ~ J
    private ArrayList<Double> theta = new ArrayList<>(); // ???

    private double F;
    private double[] phi; // 1 ~ J
    private double[] y; // ???

    private double c; // ???

    private int basicFunctionType;
    private int adjustType;

    protected RadialBasisFunctionNetwork(){
        x = new double[N + 1][P + 1]; // 1 ~ N, 1 ~ P
        m = new double[J + 1][P + 1]; // ???
        phi = new double[J + 1];
        y = new double[N + 1];

        theta.add(offset);

        double[] ww = new double[J + 1]; // 0 ~ J
        ww[0] = theta.get(0);
        w.add(ww);
    }

    protected void start(){
        train();

    }

    private void train(){
        while(cycle < maxLearningCycle){
            for(int n = 1; n <= N; n++){
                F = output(x[n]);
                E.add(0.5 * (y[n] - F) * (y[n] - F));
            }

            switch (adjustType){
                case ADJUST_W_THETA:
                    adjust_w_theta();
                    break;
                case ADJUST_ALL:
                    adjust_all();
                    break;
            }
            cycle++;
        }
    }

    private double output(double[] input){ // output: 1 ~ P
        double sum = w.get(cycle)[0];
        for(int j = 1; j <= J; j++){
            phi[j] = basicFunction(input, j);
            sum += w.get(cycle)[j] * phi[j];
        }

        return sum;
    }

    private double basicFunction(double[] input, int j){ // return <= 0 // input: 1 ~ P // j is for m[]
        double[] minus = new double[P + 1]; // 1 ~ P
        double[] minus_abs = new double[P + 1]; // 1 ~ P
        double norms = 0;
        for(int i = 1; i <= P; i++){
            minus[i] = input[i] - m[j][i];
            minus_abs[i] = Math.abs(minus[i]);
            norms += minus_abs[i] * minus_abs[i];
        }

        switch (basicFunctionType){
            case BASIC_FUNCTION_TYPE1:
                return (1 / Math.sqrt(norms + (c * c)));
            case BASIC_FUNCTION_TYPE2:
                return Math.exp(-(norms / (2 * sigma[j] * sigma[j])));
            case BASIC_FUNCTION_TYPE3:
                double[][] minus_transp = new double[minus.length][1];
                for(int i = 0; i < minus.length; i++){
                    minus_transp[i][0] = minus[i];
                }

                // ???
                double sum = 0;
                for(int i = 0; i < minus_transp.length; i++){
                    sum += (minus_transp[i][0] * minus[i]);
                }

                return Math.exp((-0.5) * sum);
            default:
                return -1;
        }
    }

    private void adjust_w_theta(){
        LMS();
        virtualCounterMatrix();
    }

    private void adjust_all(){

    }

    private void LMS(){
        adjustW();
        adjustTheta();
    }

    private void adjustW(){
        double[] ww = new double[J + 1];
        for(int j = 1; j <= J; j++){
            ww[j] = w.get(cycle)[j] + (learningRate * (y[j] - F) * phi[j]); // ???
        }
        w.add(ww);
    }

    private void adjustTheta(){
        double sum = 0;
        for(int j = 0; j < w.get(cycle).length; j++){
            sum += (learningRate * (y[j] - F)); // ???
        }
        theta.add(theta.get(cycle) + sum);
    }

    private void virtualCounterMatrix(){
        double[][] PHI = new double[N + 1][J + 1];
        for(int n = 1; n <= N; n++){
            PHI[n][0] = 1;
            for(int j = 1; j <= J; j++){
                PHI[n][j] = basicFunction(x[n], j);
            }
        }
    }
}
