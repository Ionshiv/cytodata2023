package davho884.livecellpaint;
import org.scijava.plugin.*;
import org.scijava.log.*;
import org.scijava.command.*;
import org.scijava.ItemIO;
import io.scif.services.*;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import java.io.*;
/**
 * Hello world!
 *
 */
@Plugin(type = Command.class, menuPath = "Tutorials>Open Image")








//This macro generates hyper-stacks from the image series, then saves them as hyper-tiffs

public class StackMacro implements Command{
    @Parameter
    private DatasetIOService datasetIOService;

    @Parameter
    private LogService logService;

    @Parameter 
    private File imageFile;

    @Parameter(type = ItemIO.OUTPUT)
    private Dataset image;

    public StackMacro(){
        final ImageJ ij = new ImageJ();
    }

    public void importSeq(){
        //imports TIF sequence
        return;
    }

    public void toStack(){
        
    }

    public void toHyper(){
        //Converts to TIF Hyperstack
        return;

    }

    public void subStacking(){
        //Splits the TIF hyperstack into substacks by channel
    }

    public void saveStack(){
        //saves Hyperstacks
        return;
    }

    @Override
    public void run() {
        // TODO Auto-generated method stub
        
    }
    public static void main( String[] args ) throws Exception 
    {     
        StackMacro sm = new StackMacro();


    }
    
}
