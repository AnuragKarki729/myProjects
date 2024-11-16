"use client";
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '../components/Navbar';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const TrainerPage = () => {
    const [appointments, setAppointments] = useState([]);
    const [busyDates, setBusyDates] = useState([]);
    const [showModal, setShowModal] = useState(false);
    const [newBusyDate, setNewBusyDate] = useState('');
    const router = useRouter();

    useEffect(() => {
        const currentUser = JSON.parse(localStorage.getItem('currentUser'));
        if (!currentUser || !currentUser.isTrainer) {
            router.push('/login');
        } else {
            fetchAppointments(currentUser._id);
            fetchBusyDates(currentUser._id);
        }
    }, [router]);

    const fetchAppointments = async (trainerID) => {
        const response = await fetch(`/api/appointments?trainerID=${trainerID}`);
        const data = await response.json();
        setAppointments(data);
    };

    const fetchBusyDates = async (trainerID) => {
        const response = await fetch(`/api/trainers/${trainerID}`);
        const data = await response.json();
        setBusyDates(data.busyDates);
    };

    const handleStatusUpdate = async (appointmentID, newStatus) => {
        const response = await fetch(`/api/appointments/${appointmentID}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus }),
        });

        if (response.ok) {
            const currentUser = JSON.parse(localStorage.getItem('currentUser'));
            if (currentUser && currentUser._id) {
                fetchAppointments(currentUser._id);
            }
        }
    };

    const handleAddBusyDate = async () => {
        const trainerID = JSON.parse(localStorage.getItem('currentUser'))._id;
        const response = await fetch(`/api/trainers/${trainerID}/busy-dates`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ date: newBusyDate }),
        });

        if (response.ok) {
            fetchBusyDates(trainerID);
            setShowModal(false);
            setNewBusyDate('');
        }
    };

    return (
        <>
            <Navbar pageTitle="Trainer Panel" />
            <div className="container mx-auto p-4">
                <Card className="mb-6">
                    <CardHeader>
                        <CardTitle>Your Appointments</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Client Name</TableHead>
                                    <TableHead>Appointment Date</TableHead>
                                    <TableHead>Status</TableHead>
                                    <TableHead>Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {appointments.map(appointment => (
                                    <TableRow key={appointment._id}>
                                        <TableCell>{appointment.user.username}</TableCell>
                                        <TableCell>{new Date(appointment.date).toLocaleString()}</TableCell>
                                        <TableCell>{appointment.status}</TableCell>
                                        <TableCell>
                                            <Button variant="outline" className="mr-2" onClick={() => handleStatusUpdate(appointment._id, 'Completed')}>Mark as Completed</Button>
                                            <Button variant="outline" onClick={() => handleStatusUpdate(appointment._id, 'Pending')}>Mark as Pending</Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Manage Busy Dates</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Dialog open={showModal} onOpenChange={setShowModal}>
                            <DialogTrigger asChild>
                                <Button>Add Busy Date</Button>
                            </DialogTrigger>
                            <DialogContent>
                                <DialogHeader>
                                    <DialogTitle>Add Busy Date</DialogTitle>
                                    <DialogDescription>Select a date when you&apos;re not available.</DialogDescription>
                                </DialogHeader>
                                <div className="grid gap-4 py-4">
                                    <div className="grid grid-cols-4 items-center gap-4">
                                        <Label htmlFor="busyDate" className="text-right">Busy Date</Label>
                                        <Input id="busyDate" type="date" value={newBusyDate} onChange={(e) => setNewBusyDate(e.target.value)} className="col-span-3" />
                                    </div>
                                </div>
                                <DialogFooter>
                                    <Button onClick={handleAddBusyDate}>Add Busy Date</Button>
                                </DialogFooter>
                            </DialogContent>
                        </Dialog>

                        <div className="mt-4">
                            <h3 className="text-lg font-semibold mb-2">Current Busy Dates</h3>
                            {busyDates.length > 0 ? (
                                <ul className="list-disc pl-5">
                                    {busyDates.map((date, index) => (
                                        <li key={index}>{new Date(date).toLocaleDateString()}</li>
                                    ))}
                                </ul>
                            ) : (
                                <p>No busy dates added.</p>
                            )}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </>
    );
};

export default TrainerPage;