import dbConnect from '@/lib/mongodb';
import User from '@/models/User';
import Appointment from '@/models/Appointment';
import Trainer from '@/models/Trainer'; // Import Trainer model if needed

export async function GET() {
  await dbConnect();

  try {
    // Fetch all users
    const users = await User.find();

    // For each user, fetch their appointments and populate trainer details
    const userAppointments = await Promise.all(
      users.map(async (user) => {
        // Fetch appointments for each user
        const appointments = await Appointment.find({ user: user._id }).populate('trainer').exec();

        const populatedAppointments = appointments.map(appointment => ({
          _id: appointment._id,
          trainerName: appointment.trainer ? appointment.trainer.name : 'Unknown',
          date: appointment.date,
          status: appointment.status,
          clientName: user.username
        }));

        return {
          _id: user._id,
          isAdmin: user.isAdmin,
          userName: user.username,
          email: user.email,
          appointments: populatedAppointments,
        };
      })
    );

    return new Response(JSON.stringify(userAppointments), { status: 200 });
  } catch (error) {
    return new Response(JSON.stringify({ message: 'Error fetching users and appointments', error: error.message }), { status: 500 });
  }
}
