import { NextResponse } from 'next/server';
import connectToDatabase from '@/lib/mongodb';  // Adjust the path to your connection file
import User from '@/models/User';  // Adjust the path to your model

export async function POST(request) {
  const { username, email, password } = await request.json();  // Adjust according to your request body

  try {
    await connectToDatabase();
    const newUser = new User({ username, email, password });
    await newUser.save();
    console.log("User Registered")
    return NextResponse.json({ message: 'User Registered' }, { status: 201 });
    
  } catch (err) {
    return NextResponse.json({ error: err.message }, { status: 400 });
  }
}