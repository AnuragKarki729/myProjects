import dbConnect from '@/lib/mongodb';
import Appointment from '@/models/Appointment';

export async function GET(req) {
  await dbConnect(); // Connect to MongoDB

  const { searchParams } = new URL(req.url);
  const trainerID = searchParams.get('trainerID'); // Get trainerID from query parameters

  try {
    let appointments;

    if (trainerID) {
      // Fetch appointments for a specific trainer
      appointments = await Appointment.find({ trainer: trainerID })
        .populate('trainer', 'name')
        .populate('user', 'username'); // Populate user to get username
    } else {
      // Fetch all appointments (for admins)
      appointments = await Appointment.find()
        .populate('trainer', 'name')
        .populate('user', 'username'); // Populate user to get username
    }

    return new Response(JSON.stringify(appointments), { status: 200 });
  } catch (error) {
    return new Response(JSON.stringify({ message: 'Error fetching appointments', error: error.message }), {
      status: 500,
    });
  }
}

export async function POST(req) {
  await dbConnect(); // Connect to MongoDB

  const body = await req.json();
  const { trainerID, userID, date } = body;

  const baseUrl = new URL(req.url).origin;

  const clientNameResponse = await fetch(`${baseUrl}/api/users/${userID}`);
    const clientNameData = await clientNameResponse.json();

    const clientName = clientNameData.username;
  
  console.log(trainerID, " ", userID, " ", date)

  if (!trainerID || !userID || !date) {
    return new Response(JSON.stringify({ message: 'Missing required fields' }), {
      status: 400,
    });
  }

  try {
    const appointmentDate = new Date(date);

    // Check if the trainer is already booked during the selected time
    const conflictingAppointment = await Appointment.findOne({
      trainer: trainerID,
      date: appointmentDate,
    });

    if (conflictingAppointment) {
      return new Response(
        JSON.stringify({
          message: `Trainer is already booked at ${appointmentDate.toISOString()}`,
        }),
        { status: 409 }
      );
    }

    // Create new appointment
    const newAppointment = new Appointment({
      trainer: trainerID,
      user: userID,
      date: appointmentDate,
      status: 'Confirmed',
      
    });

    await newAppointment.save();

    return new Response(JSON.stringify({ message: 'Appointment created successfully', appointment: newAppointment }), {
      status: 201,
    });
  } catch (error) {
    return new Response(JSON.stringify({ message: 'Error creating appointment', error: error.message }), {
      status: 500,
    });
  }
}
