public interface OnNeuralNetworkCallback {
    void dataReadyCallback(DataSet dataSet);
    void showResultCallback();
    void getValueCallback(int type, String string); // for JFrame: it update value, callback MLP
    void setValueCallback(int type, String string); // for JFrame: MLP update value, callback to set
}
