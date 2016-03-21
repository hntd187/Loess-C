package com.scarman.loess;


import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.analysis.interpolation.LoessInterpolator;

public class Loess {
  static {
    System.loadLibrary("loess");
  }

  public static native double[] smooth(double[] xval, double[] yval, double[] weights, double bandwidth, int iter);

  public static void main(String args[]) {
    int trials = Integer.parseInt(args[0]);
    double[] x = new double[trials];
    double[] y = new double[trials];
    double[] w = new double[trials];

    Random rg = new Random();

    for (int i = 0; i < trials; i++) {
      x[i] = rg.nextDouble();
      y[i] = rg.nextDouble();
      w[i] = 1.0;
    }

    Arrays.sort(x);
    Arrays.sort(y);

    long s = System.currentTimeMillis();
    double[] li = new LoessInterpolator().smooth(x, y, w);
    long e = System.currentTimeMillis();

    long s1 = System.currentTimeMillis();
    double[] ln = Loess.smooth(x, y, w, 0.3, 2);
    long e1 = System.currentTimeMillis();
    /*
    for (int i = 0; i < trials; i++) {
      System.out.println(ln[i] + " " + li[i]);
    }
    */
    System.out.println("Commons time: " + (e - s));
    System.out.println("Bad C Impl time: " + (e1 - s1));
  }
}
