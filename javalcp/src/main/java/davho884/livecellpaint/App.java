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

public class App implements Command
{
    @Parameter
    private DatasetIOService datasetIOService;

    @Parameter
    private LogService logService;

    @Parameter 
    private File imageFile;

    @Parameter(type = ItemIO.OUTPUT)
    private Dataset image;





    public App(String[] args){
        System.out.println( "+++INNITIATING MACRO SPIRIT+++" );
        final ImageJ ij = new ImageJ();
        ij.launch(args);
        ij.command().run(App.class, true);



    }
    public static void main( String[] args ) throws Exception 
    {     
        App app = new App(args);


    }
    @Override
    public void run() {
        try{
            image = datasetIOService.open(imageFile.getAbsolutePath());

        }catch (final IOException e) {
            logService.error(e);
        }
        
    }
}
