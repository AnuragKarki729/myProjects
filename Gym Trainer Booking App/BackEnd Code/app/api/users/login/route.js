import { NextResponse } from 'next/server';
import connectToDatabase from '@/lib/mongodb';
import User from '@/models/User'; // Adjust the path to your model
import Trainer from '@/models/Trainer'; // Import Trainer model

export async function POST(request) {
  const { email, password } = await request.json();

  try {
    await connectToDatabase();

    // First, check if the user is an admin or regular user (in User model)
    const user = await User.findOne({ email, password });  // Directly checking password (for simplicity, not recommended in production)

    if (user) {
      return NextResponse.json({ user: {
        _id: user._id,
        username:user.username,
        email: user.email,
        isAdmin: user.isAdmin, // True for admin, false for regular user
        isTrainer: false,      // Not a trainer if found in User model
      }});
    }

    // If not found in User model, check if the user is a trainer
    const trainer = await Trainer.findOne({ email, password }); // Directly checking password

    if (trainer) {
      return NextResponse.json({ user:{
        _id: trainer._id,
        name:trainer.name,
        email: trainer.email,
        isAdmin: false,         // Not an admin
        isTrainer: true,        // True for trainer
    }});
    }

    // If neither user nor trainer is found, return an error
    return NextResponse.json({ error: 'Invalid Email or Password' }, { status: 400 });

  } catch (err) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
