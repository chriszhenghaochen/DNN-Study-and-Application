package comp5216.sydney.edu.au.todolist;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Fragment1 fragment1 = new Fragment1();
        //getFragmentManager().beginTransaction().replace(R.id.mainPage, fragment1).commit();
        Fragment2 fragment2 = new Fragment2();
        getFragmentManager().beginTransaction().replace(R.id.mainPage, fragment2).commit();
    }
}
