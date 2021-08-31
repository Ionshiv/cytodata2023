package davho884.livecellpaint;
import ij.*;
import java.util.*;
import java.awt.*;
import java.io.*;
import java.lang.reflect.Array;

public class CmdMacro {



    public CmdMacro(){

    }
    public void endMsg(){
        System.out.println("+++ENDING MACRO+++");
        return;
    }
    public ImagePlus loadImg(String imgPath){
        ij.io.Opener opener = new ij.io.Opener();
        ImagePlus img = new ImagePlus();
        img = opener.openImage(imgPath);
        return img;
    }
}
