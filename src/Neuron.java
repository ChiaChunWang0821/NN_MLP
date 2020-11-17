import java.math.BigDecimal;

public class Neuron {

    protected int numOfPreviousLayer;
    protected double[] w;
    protected double[] modify_w;
    protected double v;
    protected double y;
    protected double d;
    protected double delta;
    protected double threshold;

    protected Neuron(int pre){
        numOfPreviousLayer = pre;

        w = new double[numOfPreviousLayer];
        modify_w = new double[numOfPreviousLayer];

        if(w.length != 0){
            w[0] = threshold; // pair with x0 = -1
            for(int i = 1; i < numOfPreviousLayer; i++){
                BigDecimal bigDecimal = new BigDecimal(Math.random() - 0.5);
                w[i] = bigDecimal.setScale(4, BigDecimal.ROUND_HALF_UP).doubleValue();
            }
        }
    }

    protected double errorFunction(){
        return d - y; // e
    }

    protected double activationFunction(){ // v is between -unlimited to unlimited
        return (1.0 / (1 + Math.exp(-v)));
    }

    protected double differentialActivationFunction(double num){
        return num * (1 - num);
    }
}
