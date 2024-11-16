"use client";
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '../components/Navbar';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";

const HomeScreen = () => {
  const [trainers, setTrainers] = useState([]);
  const [user, setUser] = useState(null);
  const router = useRouter();

  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem('currentUser'));
    if (!storedUser) {
      router.push('/login');
    } else {
      setUser(storedUser);
    }
  }, [router]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/trainers');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setTrainers(data);
      } catch (error) {
        console.error('Error fetching trainers:', error);
      }
    };
    fetchData();
  }, []);

  return (
    <>
      <Navbar pageTitle="Choose Trainer"/>
      <div className="container mx-auto p-4">
        <h1 className="text-3xl font-bold mb-6">Available Trainers</h1>
        <ScrollArea className="h-[calc(100vh-200px)]">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {trainers.map((trainer) => (
              <Card key={trainer._id} className="flex flex-col justify-between">
                <CardHeader>
                  <Avatar className="w-24 h-24 mx-auto mb-4">
                    <AvatarImage src={trainer.imgUrl[0]} alt={trainer.name} />
                    <AvatarFallback>{trainer.name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <CardTitle>{trainer.name}</CardTitle>
                  <CardDescription>{trainer.type} Trainer</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{trainer.description}</p>
                  <p className="mt-2 font-semibold">Price: ${trainer.price}/hour</p>
                  <p>Experience: {trainer.expYears} years</p>
                </CardContent>
                <CardFooter>
                  <Button className="w-full" onClick={() => router.push(`/book/${trainer._id}`)}>
                    Book Now
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </div>
    </>
  );
};

export default HomeScreen;