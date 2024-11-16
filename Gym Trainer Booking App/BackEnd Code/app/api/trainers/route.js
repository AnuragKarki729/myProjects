import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb'; // Ensure this path is correct
import Trainer from '@/models/Trainer'; // Import your Trainer model

export async function GET() {
  try {
    await dbConnect(); // Ensure database connection
    const trainers = await Trainer.find({}); // Use the Mongoose model to find trainers
    // console.log(trainers);
    return NextResponse.json(trainers);
  } catch (error) {
    console.error('Error fetching trainers:', error);
    return NextResponse.json({ message: error.message }, { status: 500 });
  }
}

export async function POST(req) {
  await dbConnect();

  const { name, age, expYears, phone, price, imgUrl, type, description, email, password } = await req.json();

  try {
    const newTrainer = new Trainer({
      name,
      age,
      expYears,
      phone,
      price,
      imgUrl,
      type,
      description,
      email,
      password,
    });

    await newTrainer.save();
    return new Response(JSON.stringify({ message: 'Trainer created successfully', trainer: newTrainer }), { status: 201 });
  } catch (error) {
    return new Response(JSON.stringify({ message: 'Error creating trainer', error: error.message }), { status: 500 });
  }
}









