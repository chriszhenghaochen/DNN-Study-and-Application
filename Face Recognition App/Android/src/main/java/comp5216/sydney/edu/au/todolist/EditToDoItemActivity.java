package comp5216.sydney.edu.au.todolist;

import android.Manifest;
import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.location.LocationServices;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.List;
import java.util.Locale;


public class EditToDoItemActivity extends Activity implements
		GoogleApiClient.ConnectionCallbacks, GoogleApiClient.OnConnectionFailedListener {
	public int position=0;
	private EditText etItem;
	private TextView creation_time;
	private String photo_path;
	private File destination;
	private ByteArrayOutputStream bytes;
	public final static int CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE = 1034;
	public final static int PICK_PHOTO_CODE = 1046;
	private GoogleApiClient mGoogleApiClient;
	private Location mLastLocation = null;
	private String location;


	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		
		//populate the screen using the layout
		setContentView(R.layout.activity_edit_item);
		
		//Get the data from the main screen
		String editItem = getIntent().getStringExtra("item");
		position = getIntent().getIntExtra("position",-1);
		
		// show original content in the text field
		etItem = (EditText)findViewById(R.id.etEditItem);
		etItem.setText(editItem);

		// show the creation time
		creation_time = (TextView)findViewById(R.id.creation_time);
		creation_time.setText(getIntent().getStringExtra("time"));

		// show the photo
		photo_path = getIntent().getStringExtra("photo_path");
		if (photo_path.equals("") || photo_path == null) {
			photo_path = null;
		}
		else {
			Bitmap bitmap = BitmapFactory.decodeFile(photo_path);
			((ImageView)findViewById(R.id.photo)).setImageBitmap(bitmap);
		}

		// location service
		if (ContextCompat.checkSelfPermission(this,
				Manifest.permission.ACCESS_COARSE_LOCATION)
				!= PackageManager.PERMISSION_GRANTED) {

			// No explanation needed, we can request the permission.

			ActivityCompat.requestPermissions(this,
					new String[]{Manifest.permission.ACCESS_COARSE_LOCATION},
					233);
		}
		else {
			if (mGoogleApiClient == null) {
				mGoogleApiClient = new GoogleApiClient.Builder(this)
						.addConnectionCallbacks(this)
						.addOnConnectionFailedListener(this)
						.addApi(LocationServices.API)
						.build();
			}
		}
	}

	@Override
	public void onConnected(Bundle connectionHint) {
		try {
			mLastLocation = LocationServices.FusedLocationApi.getLastLocation(
					mGoogleApiClient);
			if (mLastLocation != null) {
				Geocoder geocoder = new Geocoder(this, Locale.getDefault());
				List<Address> addresses = geocoder.getFromLocation(mLastLocation.getLatitude(),
						mLastLocation.getLongitude(), 1);
				String address = addresses.get(0).getAddressLine(0);
				String city = addresses.get(0).getLocality();
				String state = addresses.get(0).getAdminArea();
				String country = addresses.get(0).getCountryName();
				String postalCode = addresses.get(0).getPostalCode();

				location = address + ", " + city + ", " + state
						+ ", " + country + ", " + postalCode;
				((TextView)findViewById(R.id.location)).setText(location);
			}
		}
		catch (SecurityException e) {
			e.printStackTrace();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected void onStart() {
		mGoogleApiClient.connect();
		super.onStart();
	}

	protected void onStop() {
		mGoogleApiClient.disconnect();
		super.onStop();
	}

	@Override
	public void onConnectionSuspended(int i) {

	}
	@Override
	public void onConnectionFailed(ConnectionResult result) {

	}


	// select the image
	public void selectImage(View view) {
		final CharSequence[] items = { "Take Photo", "Choose from Library",
				"Cancel" };
		AlertDialog.Builder builder = new AlertDialog.Builder(this);

		// customized title
		TextView title = new TextView(this);
		title.setText("Add Photo");
		title.setGravity(Gravity.CENTER_HORIZONTAL | Gravity.CENTER_VERTICAL);
		title.setBackgroundColor(Color.DKGRAY);
		title.setPadding(10, 10, 10, 10);
		title.setTextColor(Color.WHITE);
		title.setTextSize(20);
		builder.setCustomTitle(title);

		builder.setItems(items, new DialogInterface.OnClickListener() {
			@Override
			public void onClick(DialogInterface dialog, int item) {
				// boolean result=Utility.checkPermission(getActivity());
				if (items[item].equals("Take Photo")) {
					onTakePhotoClick();
				} else if (items[item].equals("Choose from Library")) {
					onLoadPhotoClick();
				} else if (items[item].equals("Cancel")) {
					dialog.dismiss();
				}
			}
		});
		builder.show();
	}

	// get image in gallery
	private void onTakePhotoClick()
	{
		Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
		startActivityForResult(intent, CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE);
	}

	// get image from photo
	private void onLoadPhotoClick()
	{
		Intent intent = new Intent();
		intent.setType("image/*");
		intent.setAction(Intent.ACTION_GET_CONTENT);//
		startActivityForResult(Intent.createChooser(intent, "Select File"), PICK_PHOTO_CODE);
	}

	@Override
	public void onActivityResult(int requestCode, int resultCode, Intent data) {
		ImageView photo = (ImageView)findViewById(R.id.photo);

		// deal with picked photo
		if (requestCode == PICK_PHOTO_CODE) {
			if (resultCode == RESULT_OK) {
				Uri photoUri = data.getData();
				Bitmap selectedImage;
				try {
					selectedImage = MediaStore.Images.Media.getBitmap(
							this.getContentResolver(), photoUri);
					// Load the selected image into a preview
					photo.setImageBitmap(selectedImage);

					photo_path = ImageFilePath.getPath(this, photoUri);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}

		// deal with taken photo
		if (requestCode == CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE) {
			if (resultCode == RESULT_OK) {
				Bitmap thumbnail = (Bitmap) data.getExtras().get("data");
				bytes = new ByteArrayOutputStream();
				thumbnail.compress(Bitmap.CompressFormat.JPEG, 100, bytes);

				destination = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
						System.currentTimeMillis() + ".jpg");

				try {

					// request write permission
					// Here, thisActivity is the current activity
					if (ContextCompat.checkSelfPermission(this,
							Manifest.permission.WRITE_EXTERNAL_STORAGE)
							!= PackageManager.PERMISSION_GRANTED) {

						// No explanation needed, we can request the permission.

						ActivityCompat.requestPermissions(this,
									new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
									666);
					}
					else {
						destination.createNewFile();
						FileOutputStream fo = new FileOutputStream(destination);
						fo.write(bytes.toByteArray());
						fo.close();
					}
				} catch (Exception e) {
					e.printStackTrace();
				}

				photo.setImageBitmap(thumbnail);
				photo_path = destination.getAbsolutePath();
				Toast.makeText(this, photo_path,
						Toast.LENGTH_SHORT).show();
			}
			else {
				Toast.makeText(this, "Picture wasn't taken!",
						Toast.LENGTH_SHORT).show();
			}
		}
	}

	@Override
	public void onRequestPermissionsResult(int requestCode,
										   String permissions[], int[] grantResults) {
		switch (requestCode) {
			case 666: {
				// If request is cancelled, the result arrays are empty.
				if (grantResults.length > 0
						&& grantResults[0] == PackageManager.PERMISSION_GRANTED) {
					try {
						destination.createNewFile();
						FileOutputStream fo = new FileOutputStream(destination);
						fo.write(bytes.toByteArray());
						fo.close();
					}
					catch (Exception e) {

					}

				} else {

					// permission denied, boo! Disable the
					// functionality that depends on this permission.
				}
				return;
			}

			case 233: {
				if (mGoogleApiClient == null) {
					mGoogleApiClient = new GoogleApiClient.Builder(this)
							.addConnectionCallbacks(this)
							.addOnConnectionFailedListener(this)
							.addApi(LocationServices.API)
							.build();
				}
			}

			// other 'case' lines to check for other
			// permissions this app might request
		}
	}

	// when user clicks "submit"
	public void onSubmit(View v) {
		// if something wrong
		if (photo_path == null) {
			AlertDialog.Builder builder = new AlertDialog.Builder(this);
			builder.setTitle("").setMessage("Please take a photo or select from gallery");
			builder.create().show();
		}
		// if nothing wrong
		else {
			etItem = (EditText) findViewById(R.id.etEditItem);

			// Prepare data intent for sending it back
			Intent data = new Intent();

			// Pass relevant data back as a result
			data.putExtra("item", etItem.getText().toString());
			data.putExtra("position", position);
			data.putExtra("photo_path", photo_path);
			data.putExtra("location", location);

			// get current time and send back to main activity
			Calendar c = Calendar.getInstance();
			SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
			String formattedDate = df.format(c.getTime());
			data.putExtra("time", formattedDate);

			// Activity finished ok, return the data
			setResult(RESULT_OK, data); // set result code and bundle data for response
			finish(); // closes the activity, pass data to parent
		}
	}

	// when user clicks "cancel"
	public void onCancel(View v) {
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setTitle("")
				.setMessage(R.string.dialog_cancel_message)
				.setPositiveButton(R.string.yes, new DialogInterface.OnClickListener() {
					public void onClick(DialogInterface dialog, int id) {
						// Prepare data intent for sending it back
						Intent data = new Intent();

						// Activity finished ok, return the data
						setResult(RESULT_CANCELED, data); // set result code and bundle data for response
						finish(); // closes the activity, pass data to parent
					}
				})
				.setNegativeButton(R.string.no, new DialogInterface.OnClickListener() {
					public void onClick(DialogInterface dialog, int id) {
						// user cancelled the dialog
						// nothing happens
					}
				});

		builder.create().show();
	}
}
