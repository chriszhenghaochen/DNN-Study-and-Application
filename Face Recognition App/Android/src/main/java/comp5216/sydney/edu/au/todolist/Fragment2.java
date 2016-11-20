package comp5216.sydney.edu.au.todolist;

import android.app.AlertDialog;
import android.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.text.Html;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CompoundButton;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhlu2106 on 16/09/2016.
 */
public class Fragment2 extends Fragment {
    ListView listview;
    GridView gridView;
    ArrayList<Item> items;
    CustomAdapter itemsAdapter;
    GridAdapter gridAdapter;
    public final int EDIT_ITEM_REQUEST_CODE = 647;
    private int display = 1;

    public View onCreateView(
            LayoutInflater inflater,
            ViewGroup container,
            Bundle savedInstanceState) {
        final View view = inflater.inflate(R.layout.items, container, false);

        // create arraylist of string
        readItemsFromDatabase();

        // create an adapter and connect to list
        itemsAdapter = new CustomAdapter(getActivity(), items);
        listview = (ListView)view.findViewById(R.id.listview);
        listview.setAdapter(itemsAdapter);

        // create adapter for grid view
        gridAdapter = new GridAdapter(getActivity(), items);
        gridView = (GridView)view.findViewById(R.id.gridview);
        gridView.setAdapter(gridAdapter);

        setupListViewListener();
        view.findViewById(R.id.btnAddItem).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onAddItemClick(v);
            }
        });
        ((Switch)view.findViewById(R.id.switch1)).setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                // list view
                if (isChecked) {
                    display = 2;
                    view.findViewById(R.id.listview).setVisibility(View.VISIBLE);
                    view.findViewById(R.id.gridview).setVisibility(View.GONE);
                    itemsAdapter.notifyDataSetChanged();
                }
                // grid view
                else {
                    display = 1;
                    view.findViewById(R.id.listview).setVisibility(View.GONE);
                    view.findViewById(R.id.gridview).setVisibility(View.VISIBLE);
                    itemsAdapter.notifyDataSetChanged();
                }
            }
        });

        return view;
    }

    public class CustomAdapter extends ArrayAdapter {
        public CustomAdapter(Context context, ArrayList<Item> users) {super(context, 0, users);}

        @Override
        public Item getItem(int arg0)
        {
            return items.get(arg0);
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            convertView = LayoutInflater.from(getContext()).inflate(R.layout.item, parent, false);
            if (display == 2) {
                Item item = getItem(position);
                TextView todo = (TextView) convertView.findViewById(R.id.todo);
                TextView creation_time = (TextView) convertView.findViewById(R.id.creation_time);
                TextView location = (TextView) convertView.findViewById(R.id.location);
                todo.setText(Html.fromHtml("<b>Title: </b>" + item.todo));
                creation_time.setText(Html.fromHtml("<b>Created: </b>" + item.creation_time));
                location.setText(Html.fromHtml("<b>Location: </b>" + item.location));

                // photo
                BitmapFactory.Options o = new BitmapFactory.Options();
                o.inSampleSize = 5;
                Bitmap bitmap = BitmapFactory.decodeFile(item.photo_path, o);
                ((ImageView)convertView.findViewById(R.id.img1)).setImageBitmap(bitmap);
            }

            return convertView;
        }
    }

    public class GridAdapter extends ArrayAdapter {
        public GridAdapter(Context context, ArrayList<Item> users) {super(context, 0, users);}

        @Override
        public Item getItem(int arg0)
        {
            return items.get(arg0);
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            convertView = LayoutInflater.from(getContext()).inflate(R.layout.grid_item, parent, false);
            if (display == 1) {
                Item item = getItem(position);
                convertView = LayoutInflater.from(getContext()).inflate(R.layout.grid_item, parent, false);
                TextView todo = (TextView) convertView.findViewById(R.id.todo);
                todo.setText(item.todo);

                BitmapFactory.Options o = new BitmapFactory.Options();
                o.inSampleSize = 5;
                Bitmap bitmap = BitmapFactory.decodeFile(item.photo_path, o);
                ((ImageView)convertView.findViewById(R.id.img1)).setImageBitmap(bitmap);
            }

            return convertView;
        }
    }

    public void onAddItemClick(View view) {
        Intent intent = new Intent(getActivity(), EditToDoItemActivity.class);
        if (intent != null) {
            // put "extras" into the bundle for access in the dit activity
            intent.putExtra("item", "");
            intent.putExtra("position", -1);
            intent.putExtra("photo_path", "");
            // bring up the second activity
            startActivityForResult(intent, EDIT_ITEM_REQUEST_CODE);
            itemsAdapter.notifyDataSetChanged();
        }
    }

    private void setupListViewListener() {
        listview.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
            public boolean onItemLongClick(AdapterView<?> parent, View view, final int position, long rowId) {
                Log.i("MainActivity", "Long Clicked item" + position);
                AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
                builder.setTitle(R.string.dialog_delete_title)
                        .setMessage(R.string.dialog_delete_message)
                        .setPositiveButton(R.string.yes, new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                // delete an item
                                items.remove(position);
                                itemsAdapter.notifyDataSetChanged();

                                saveItemsToDatabase();
                            }
                        })
                        .setNegativeButton(R.string.no, new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                // user cancelled the dialog
                                // nothing happens
                            }
                        });

                builder.create().show();
                return true;
            }
        });
        listview.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long rowId) {
                String updateItem = itemsAdapter.getItem(position).todo;
                Log.i("MainActivity", "Clicked item" + position + ": " + updateItem);

                Intent intent = new Intent(getActivity(), EditToDoItemActivity.class);
                if (intent != null) {
                    // put "extras" into the bundle for access in the dit activity
                    intent.putExtra("item", updateItem);
                    intent.putExtra("time", itemsAdapter.getItem(position).creation_time);
                    intent.putExtra("position", position);
                    intent.putExtra("photo_path", itemsAdapter.getItem(position).photo_path);
                    // bring up the second activity
                    startActivityForResult(intent, EDIT_ITEM_REQUEST_CODE);
                    itemsAdapter.notifyDataSetChanged();
                }
            }
        });
        gridView.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
            public boolean onItemLongClick(AdapterView<?> parent, View view, final int position, long rowId) {
                Log.i("MainActivity", "Long Clicked item" + position);
                AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
                builder.setTitle(R.string.dialog_delete_title)
                        .setMessage(R.string.dialog_delete_message)
                        .setPositiveButton(R.string.yes, new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                // delete an item
                                items.remove(position);
                                itemsAdapter.notifyDataSetChanged();

                                saveItemsToDatabase();
                            }
                        })
                        .setNegativeButton(R.string.no, new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                // user cancelled the dialog
                                // nothing happens
                            }
                        });

                builder.create().show();
                return true;
            }
        });
        gridView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long rowId) {
                String updateItem = itemsAdapter.getItem(position).todo;
                Log.i("MainActivity", "Clicked item" + position + ": " + updateItem);

                Intent intent = new Intent(getActivity(), EditToDoItemActivity.class);
                if (intent != null) {
                    // put "extras" into the bundle for access in the dit activity
                    intent.putExtra("item", updateItem);
                    intent.putExtra("time", itemsAdapter.getItem(position).creation_time);
                    intent.putExtra("position", position);
                    intent.putExtra("photo_path", itemsAdapter.getItem(position).photo_path);
                    // bring up the second activity
                    startActivityForResult(intent, EDIT_ITEM_REQUEST_CODE);
                    itemsAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    // when receive message
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == EDIT_ITEM_REQUEST_CODE) {
            if (resultCode == getActivity().RESULT_OK) {
                // extract name value from result extras
                String editedItem = data.getExtras().getString("item");
                String creation_time = data.getExtras().getString("time");
                String photo_path = data.getExtras().getString("photo_path");
                String location = data.getExtras().getString("location");
                int position = data.getIntExtra("position", -1);

                if (position == -1) {
                    items.add(new Item(editedItem, creation_time, photo_path, location));
                    Log.i("New Item in list: ", editedItem + ", position:" + position);

                    Toast.makeText(getActivity(), "New: " + editedItem, Toast.LENGTH_SHORT).show();
                }
                else {
                    items.get(position).todo = editedItem;
                    items.get(position).creation_time = creation_time;
                    items.get(position).photo_path = photo_path;
                    Log.i("Updated Item in list: ", editedItem + ", position:" + position);

                    Toast.makeText(getActivity(), "Updated: " + editedItem, Toast.LENGTH_SHORT).show();
                }

                itemsAdapter.notifyDataSetChanged();
                saveItemsToDatabase();
            }
        }
    }

    // read from database
    private void readItemsFromDatabase() {
        // read items from database
        List<ToDoItem> itemsFromORM = ToDoItem.listAll(ToDoItem.class);
        items = new ArrayList<Item>();
        if (itemsFromORM != null && itemsFromORM.size() > 0) {
            for (ToDoItem item : itemsFromORM) {
                items.add(new Item(item.todo, item.time, item.photo_path, item.location));
            }
        }
    }

    // save to database
    private void saveItemsToDatabase() {
        ToDoItem.deleteAll(ToDoItem.class);
        for (Item instance : items) {
            ToDoItem item = new ToDoItem(instance.todo, instance.creation_time, instance.photo_path,
                    instance.location);
            item.save();
        }
    }
}
