package comp5216.sydney.edu.au.todolist;

import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import com.facebook.CallbackManager;
import com.facebook.FacebookCallback;
import com.facebook.FacebookException;
import com.facebook.FacebookSdk;
import com.facebook.login.LoginResult;
import com.facebook.login.widget.LoginButton;

/**
 * Created by zhlu2106 on 16/09/2016.
 */
public class Fragment1 extends Fragment {
    LoginButton loginButton;
    CallbackManager callbackManager;

    public View onCreateView(
            LayoutInflater inflater,
            ViewGroup container,
            Bundle savedInstanceState) {
        FacebookSdk.sdkInitialize(getActivity());
        View view = inflater.inflate(R.layout.login, container, false);

        callbackManager = CallbackManager.Factory.create();

        loginButton = (LoginButton) view.findViewById(R.id.login_button);
        // If using in a fragment
        loginButton.setFragment(this);
        // Other app specific specialization

        // Callback registration
        loginButton.registerCallback(callbackManager, new FacebookCallback<LoginResult>() {
            @Override
            public void onSuccess(LoginResult loginResult) {
                Toast.makeText(getActivity(), "Facebook Login successfully!", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onCancel() {
                Toast.makeText(getActivity(), "Facebook Login cancelled!", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onError(FacebookException exception) {
                Toast.makeText(getActivity(), "Facebook Login Error!", Toast.LENGTH_SHORT).show();
            }
        });

        return view;
    }
}
