package davho884.livecellpaint;
import ij.*;
import java.util.*;
import java.awt.*;
import java.io.*;
import java.lang.reflect.Array;

//this macro generates the automated tracking

public class TrackMacro {



    public TrackMacro(){

    }
    public void endMsg(){
        System.out.println("+++ENDING MACRO+++");
        return;
    }
    //loads TIF stack
    public ImagePlus loadImg(String imgPath){
        ij.io.Opener opener = new ij.io.Opener();
        ImagePlus img = new ImagePlus();
        img = opener.openImage(imgPath);
        return img;
    }
}
