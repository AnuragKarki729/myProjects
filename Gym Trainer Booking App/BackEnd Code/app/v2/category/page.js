"use client";
import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import Link from "next/link";

import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import ListItemAvatar from "@mui/material/ListItemAvatar";
import Avatar from "@mui/material/Avatar";
import ImageIcon from "@mui/icons-material/Image";
import WorkIcon from "@mui/icons-material/Work";
import BeachAccessIcon from "@mui/icons-material/BeachAccess";

export default function Home() {
  const [category, setCategory] = useState([]);
  const [editMode, setEditMode] = useState(false);

  const { register, handleSubmit, reset, setValue } = useForm();

  const APIBASE = process.env.NEXT_PUBLIC_API_URL
  console.log(`${APIBASE}/category`)

  const startEdit = (category) => {
    // TODO
    category.id = category._id
    reset(category)
    setEditMode(true)
  }

  async function fetchCategory() {
    const data = await fetch(`${APIBASE}/category`);
    const c = await data.json();
    setCategory(c);
  }

  useEffect(() => {
    fetchCategory();
  }, []);

  function handleCategoryFormSubmit(data) {
    if (editMode) {
      // data.id = data._id
      fetch(`${APIBASE}/category`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      }).then(() => {
        reset({ name: '', order: '' })
        fetchCategory()
      });
      return
    }
    fetch(`${APIBASE}/category`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    }).then(() => {
      reset({ name: '', order: '' })
      fetchCategory()
    });
  }

  return (
    <main>
      <form onSubmit={handleSubmit(handleCategoryFormSubmit)}>
        <input type="hidden" {...register("id")} />
        <div className="grid grid-cols-2 gap-4 w-fit m-4 p-4 border border-gray-900">
          <div>Name:</div>
          <div>
            <input
              name="name"
              type="text"
              {...register("name", { required: true })}
              className="border border-gray-600 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
            />
          </div>
          <div>Order:</div>
          <div>
            <input
              name="order"
              type="number"
              {...register("order", { required: true, defaultValue: 0 })}
              className="border border-gray-600 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
            />
          </div>
          <div className="col-span-2 text-right">
            {editMode ?
              <input
                type="submit"
                value="Update"
                className="bg-blue-800 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full"
              />

              :
              <input
                type="submit"
                value="Add"
                className="bg-green-800 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-full"
              />
            }
            {
              editMode &&
              <button
                onClick={() => {
                  reset({ name: '', order: '' })
                  setEditMode(false)
                }}
                className="ml-2 bg-gray-800 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-full"
              >Cancel</button>
            }
          </div>
        </div>
      </form>
      <div className="mx-4">
        <h1>Category ({category.length})</h1>
        <ul>
          {category.map((category) => (
            <li key={category._id}>
              <button onClick={() => startEdit(category)}>📝</button>{' '}
              <Link href={`/product/category/${category._id}`} className="text-red-600">
              {category.name} [{category.order}]

              </Link>
            </li>
          ))}
        </ul>
      </div>
    </main>
  );
}
