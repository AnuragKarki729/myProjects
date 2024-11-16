import dbConnect from '@/lib/mongodb'; // Ensure this path is correct
import Trainer from '@/models/Trainer';

export async function POST(request) {
  const { selectedDate } = await request.json();

  const selectedStartDate = new Date(selectedDate);
  const selectedEndDate = new Date(selectedDate);
  selectedEndDate.setHours(selectedEndDate.getHours() + 1);

  try {
    await dbConnect();

    const availableTrainers = await Trainer.find({
      $nor: [
        {
          currentAppointments: {
            $elemMatch: {
              date: {
                $gte: selectedStartDate,
                $lt: selectedEndDate,
              },
            },
          },
        },
      ],
    });

    return new Response(JSON.stringify(availableTrainers), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error fetching available trainers:', error);
    return new Response(
      JSON.stringify({ message: 'Error fetching trainers' }),
      { status: 500 }
    );
  }
}
