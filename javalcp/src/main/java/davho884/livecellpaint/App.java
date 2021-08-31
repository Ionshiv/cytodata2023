package davho884.livecellpaint;

/**
 * Hello world!
 *
 */
public class App 
{
    public App(){
        System.out.println( "+++INNITIATING TRACKING MACRO+++" );
        CmdMacro runMacro = new CmdMacro();
        runMacro.endMsg();

    }
    public static void main( String[] args )
    {
        App newRun = new App();

    }
}
