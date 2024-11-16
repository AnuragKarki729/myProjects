"use client";
import { useRouter } from 'next/navigation';
import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import dayjs from 'dayjs';
import Navbar from '@/app/components/Navbar';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

function BookScreen() {
    const { trainerID } = useParams();
    const [trainer, setTrainer] = useState(null);
    const [appointments, setAppointments] = useState([]);
    const [selectedDate, setSelectedDate] = useState(null);
    const [selectedTime, setSelectedTime] = useState(null);
    const [availabilityMessage, setAvailabilityMessage] = useState("");
    const [bookingSuccess, setBookingSuccess] = useState(false);
    const [userID, setUserID] = useState(null);
    const [username, setUsername] = useState("");
    const router = useRouter();

    useEffect(() => {
        const fetchData = async () => {
            try {
                const trainerResponse = await fetch(`/api/trainers/${trainerID}`);
                const trainerData = await trainerResponse.json();
                setTrainer(trainerData);

                const appointmentsResponse = await fetch(`/api/appointments?trainerID=${trainerID}`);
                const appointmentsData = await appointmentsResponse.json();
                setAppointments(appointmentsData);
            } catch (error) {
                console.error('Error fetching trainer or appointments:', error);
            }
        };

        fetchData();

        const user = JSON.parse(localStorage.getItem('currentUser'));
        if (user?.username && user?._id) {
            setUserID(user._id);
            setUsername(user.username);
        }
    }, [trainerID]);

    const handleDateChange = (date) => {
        setSelectedDate(date);
        setSelectedTime(null);
        setAvailabilityMessage("");
    };

    const handleTimeChange = (time) => {
        setSelectedTime(time);
        const selectedDateTime = dayjs(selectedDate).hour(parseInt(time.split(':')[0])).minute(0);
        checkAvailability(selectedDateTime);
    };

    const checkAvailability = (dateTime) => {
        let isBooked = false;
        for (
                    let
                    i = 0; i < appointments.length; i++) {
                    if
                        (
                        dayjs
                            (appointments[i].date).
                            isSame
                            (dateTime,
                                'minute'
                            )) {
                                isBooked = true;
                        // Found a booking at this time
                        break
                        ;
                        // Exit the loop early since we found a match
                    }
                }
                setAvailabilityMessage
                    (isBooked ?
                        `Trainer busy on
${dateTime.format(
                            'YYYY-MM-DD HH:mm'
                        )}
`
                        :
                        `Trainer available on
${dateTime.format(
                            'YYYY-MM-DD HH:mm'
                        )}
`
                    );
                return
                isBooked;
                // Return whether the time is booked or not
            };

    const handleBooking = async () => {
        if (!selectedDate || !selectedTime) {
            alert("Please select a date and time.");
            return;
        }

        if (!userID) {
            alert("User not logged in.");
            return;
        }

        const selectedDateTime = dayjs(selectedDate).hour(parseInt(selectedTime.split(':')[0])).minute(0);

        try {
            const response = await fetch('/api/appointments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trainerID,
                    userID,
                    date: selectedDateTime.toISOString(),
                }),
            });

            if (response.ok) {
                setBookingSuccess(true);
                setAvailabilityMessage("Booking successful!");
                router.push('/appointments');
            } else {
                const result = await response.json();
                setAvailabilityMessage(result.message || "Error booking the trainer.");
            }
        } catch (error) {
            console.error("Error creating appointment:", error);
            setAvailabilityMessage("Error booking the trainer.");
        }
    };

    const generateTimeSlots = () => {
        const slots = [];
        for (let i = 5; i < 22; i++) {
            slots.push(`${i.toString().padStart(2, '0')}:00`);
        }
        return slots;
    };

    return (
        <div>
            <Navbar pageTitle="Book Trainer" />
            <div className="container mx-auto p-4">
                {trainer && username ? (
                    <Card className="w-[350px] mx-auto">
                        <CardHeader>
                            <CardTitle>Trainer: {trainer.name}</CardTitle>
                            <CardDescription>{trainer.description}</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <p className="font-bold mb-2">Price: ${trainer.price}</p>
                            <p className="mb-4"><strong>User:</strong> {username}</p>
                            <div className="space-y-4">
                                <div>
                                    <Label>Select Date</Label>
                                    <Calendar
                                        mode="single"
                                        selected={selectedDate}
                                        onSelect={handleDateChange}
                                        disabled={(date) => date < dayjs().startOf('day').toDate()}
                                        className="rounded-md border"
                                    />
                                </div>
                                {selectedDate && (
                                    <div>
                                        <Label>Select Time</Label>
                                        <Select onValueChange={handleTimeChange}>
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select time" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {generateTimeSlots().map((time) => (
                                                    <SelectItem key={time} value={time}>{time}</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    </div>
                                )}
                            </div>
                            {availabilityMessage && (
                                <p className="mt-4 text-sm">{availabilityMessage}</p>
                            )}
                        </CardContent>
                        <CardFooter>
                            <Button
                                onClick={handleBooking}
                                disabled={!selectedDate || !selectedTime || bookingSuccess}
                                className="w-full"
                            >
                                {bookingSuccess ? "Booking Confirmed" : "Book Appointment"}
                            </Button>
                        </CardFooter>
                    </Card>
                ) : (
                    <p>Loading trainer details...</p>
                )}
            </div>
        </div>
    );
}

export default BookScreen;