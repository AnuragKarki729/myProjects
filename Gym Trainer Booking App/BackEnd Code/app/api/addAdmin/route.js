import dbConnect from '@/lib/mongodb';
import User from '@/models/User';

// Function to handle POST requests
export async function POST(req) {
  await dbConnect(); // Connect to the MongoDB database

  try {
    // Check if the admin already exists
    const existingAdmin = await User.findOne({ email: 'admin1@admin.com' });

    if (existingAdmin) {
      return new Response(JSON.stringify({ message: 'Admin already exists' }), { status: 400 });
    }

    // Create new admin
    const newAdmin = new User({
      username: 'admin1',
      email: 'admin1@admin.com',
      password: '123', // For simplicity, no hashing. In production, hash passwords.
      isAdmin: true,
      isTrainer:false
    });

    await newAdmin.save();

    return new Response(JSON.stringify({ message: 'Admin added successfully' }), { status: 201 });
  } catch (error) {
    return new Response(JSON.stringify({ message: 'Error adding admin', error: error.message }), { status: 500 });
  }
}
